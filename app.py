"""CreditOne HK Risk Terminal - Streamlit Dashboard.

This module provides a Streamlit-based web dashboard for SME credit risk
assessment, featuring real-time market data, SHAP-based model explanations,
and stress testing capabilities.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sme_credit_explainability import (
    generate_synthetic_sme_data,
    train_model_and_explain,
    generate_risk_report_for_company,
)

# Page Config
st.set_page_config(page_title="CreditOne | HK Risk Terminal", layout="wide", page_icon="🏙️")

# CSS & Styling
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .stMetric { background-color: white; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .insight-box-risk { background-color: #ffebee; border-left: 5px solid #d32f2f; padding: 15px; border-radius: 4px; color: #b71c1c; }
    .insight-box-safe { background-color: #e8f5e9; border-left: 5px solid #2e7d32; padding: 15px; border-radius: 4px; color: #1b5e20; }
    .report-container { background-color: white; padding: 30px; border: 1px solid #ccc; font-family: 'Times New Roman', serif; margin-top: 20px; }
    .report-table { width: 100%; border-collapse: collapse; font-family: 'Arial', sans-serif; font-size: 0.9rem; }
    .report-table th { background-color: #f0f0f0; border: 1px solid #999; padding: 8px; text-align: left; color: #333; }
    .report-table td { border: 1px solid #999; padding: 8px; vertical-align: top; }
</style>
""", unsafe_allow_html=True)

# Ticker mapping for demo companies
REAL_TICKER_MAP: Dict[str, str] = {
    "HK_00000": "700.HK",   # Tencent
    "HK_00001": "5.HK",     # HSBC
    "HK_00002": "1299.HK",  # AIA
    "HK_00003": "3690.HK",  # Meituan
    "HK_00004": "9988.HK",  # Alibaba
    "HK_00005": "388.HK"    # HKEX
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_real_market_data_safe(
    ticker_symbol: str
) -> Tuple[bool, pd.DataFrame, Dict[str, Any], str, str]:
    """Fetch market data from Yahoo Finance with caching and retry logic.

    Attempts to fetch real market data up to 3 times. If all attempts fail,
    returns simulated mock data as a fallback.

    Args:
        ticker_symbol: Yahoo Finance ticker symbol (e.g., "700.HK").

    Returns:
        Tuple containing:
            - is_real: True if real data was fetched, False if using mock data.
            - hist: DataFrame with historical price data (Close, Open, High, Low).
            - fin_data: Dict with financial metrics (Revenue, Net Income, etc.).
            - currency: Currency code (e.g., "HKD").
            - name: Company name or ticker symbol.
    """
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period="6mo")
            if hist.empty:
                raise ValueError("Empty")

            info = stock.info
            fin_data: Dict[str, Any] = {
                "Revenue": info.get("totalRevenue", 0),
                "Net Income": info.get("netIncomeToCommon", 0),
                "Total Debt": info.get("totalDebt", 0),
                "Cash": info.get("totalCash", 0)
            }
            return True, hist, fin_data, info.get('currency', 'HKD'), info.get('longName', ticker_symbol)
        except Exception:
            time.sleep(0.5)

    # Fallback to simulated data
    dates = pd.date_range(end=datetime.today(), periods=90)
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 90))
    hist_mock = pd.DataFrame(
        {'Close': prices, 'Open': prices, 'High': prices, 'Low': prices},
        index=dates
    )
    mock_fin: Dict[str, Any] = {
        "Revenue": 5e10,
        "Net Income": 1e10,
        "Total Debt": 2e10,
        "Cash": 1.5e10
    }
    return False, hist_mock, mock_fin, "HKD", f"{ticker_symbol} (Simulated)"


def fmt_financial_value(v: Any) -> str:
    """Format financial value for display.

    Args:
        v: Numeric value to format, or None.

    Returns:
        Formatted string (e.g., "1.50 B" for billions, "N/A" for None/zero).
    """
    if v is None or v == 0:
        return "N/A"
    return f"{v/1e9:.2f} B" if v > 1e9 else f"{v:,.0f}"


def main() -> None:
    """Main entry point for the Streamlit dashboard.

    Initializes the risk engine, renders UI components, and handles
    user interactions for stress testing and company selection.
    """
    # Header
    c1, c2 = st.columns([1, 5])
    with c1:
        st.markdown("# 🏙️ **CreditOne**")
    with c2:
        st.caption("HK Global Risk Terminal | v5.3 Stable Release")
    st.markdown("---")

    # Init Data
    if 'data_generated' not in st.session_state:
        with st.spinner('Initializing Risk Engine...'):
            df = generate_synthetic_sme_data()
            clf, X_test_df, shap_values, explainer = train_model_and_explain(df)
            st.session_state.update({
                'df': df,
                'X_test_df': X_test_df,
                'shap_values': shap_values,
                'data_generated': True
            })

    X_test_df: pd.DataFrame = st.session_state['X_test_df']
    shap_values = st.session_state['shap_values']

    # Sidebar Controls
    st.sidebar.header("🎛️ Stress Lab")

    shock_rev_int: int = st.sidebar.slider("Revenue Shock (%)", -50, 20, 0, 5)
    shock_rev: float = shock_rev_int / 100.0

    shock_vol: float = st.sidebar.slider("Volatility Multiplier (x)", 1.0, 3.0, 1.0, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Client Search")

    demo_ids: List[str] = list(REAL_TICKER_MAP.keys())
    selected_company: str = st.sidebar.selectbox("Select Entity ID:", demo_ids)
    real_ticker: str = REAL_TICKER_MAP.get(selected_company, "700.HK")

    is_real, hist, fin_data, currency, name = fetch_real_market_data_safe(real_ticker)

    if is_real:
        st.sidebar.success(f"🟢 Online: {real_ticker}")
    else:
        st.sidebar.warning(f"🟠 Offline: {real_ticker}")

    # Core Calc
    try:
        row = X_test_df[X_test_df['company_id'] == selected_company].iloc[0]
        idx: int = int(X_test_df.index[X_test_df['company_id'] == selected_company].tolist()[0])
    except (IndexError, KeyError):
        row = X_test_df.iloc[0]
        idx = 0

    # Stress Logic
    base_pd: float = np.clip(row['predicted_default_prob'], 0.001, 0.999)
    logit_change: float = (shock_rev * -5.0) + ((shock_vol - 1.0) * 1.5)
    new_pd: float = 1 / (1 + np.exp(-(np.log(base_pd / (1 - base_pd)) + logit_change)))

    str_rev: float = row['revenue_growth'] * (1 + shock_rev)
    str_vol: float = row['cash_flow_volatility'] * shock_vol

    # Main Dashboard
    col1, col2 = st.columns([1, 2.5])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_pd * 100,
            title={'text': "Prob. Default (PD)"},
            delta={
                'reference': base_pd * 100,
                'increasing': {'color': "red"},
                'decreasing': {'color': "green"}
            },
            gauge={
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 5], 'color': '#66bb6a'},
                    {'range': [5, 20], 'color': '#ffa726'},
                    {'range': [20, 100], 'color': '#ef5350'}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Entity: **{name}**")

    with col2:
        st.markdown("### 📊 Key Financials (Stress Input)")
        k1, k2, k3 = st.columns(3)
        k1.metric("Revenue Growth", f"{str_rev:.1%}", delta=f"{shock_rev:.0%} Shock", delta_color="inverse")
        k2.metric("Debt-to-Asset", f"{row['debt_to_asset_ratio']:.1%}", help="Threshold: 60%")
        k3.metric("CF Volatility", f"{str_vol:.2f}", delta=f"{shock_vol:.1f}x Amp", delta_color="inverse")

        # Smart Insight
        insight_class: str = "insight-box-safe"
        insight_title: str = "✅ Stability Assessment"
        insight_text: str = "All key risk metrics are within healthy ranges. The company demonstrates strong resilience against stress scenarios."

        risks: List[str] = []
        if str_rev < 0:
            risks.append(f"Revenue is contracting ({str_rev:.1%}).")
        if row['debt_to_asset_ratio'] > 0.6:
            risks.append(f"Leverage is critically high ({row['debt_to_asset_ratio']:.1%}).")
        if str_vol > 1.5:
            risks.append(f"Cash flow is highly volatile ({str_vol:.2f}x).")
        if row['past_default'] > 0.5:
            risks.append("History of default detected.")

        if risks:
            insight_class = "insight-box-risk"
            insight_title = "⚠️ Critical Risk Factor(s)"
            insight_text = " ".join(risks) + " These factors significantly elevate the Probability of Default under stress."

        st.markdown(f"""
        <div class="{insight_class}">
            <strong>{insight_title}</strong><br>
            {insight_text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Live Market Data Tab
    st.subheader("🧬 Market & Model Data")
    tab1, tab2 = st.tabs(["📈 Market Data (HKEX)", "📊 Model Attribution"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            price_fig = go.Figure(data=[
                go.Scatter(x=hist.index, y=hist['Close'], line=dict(color='#1976d2', width=2))
            ])
            price_fig.update_layout(
                title=f"Price Trend: {name}",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                template="plotly_white"
            )
            st.plotly_chart(price_fig, use_container_width=True)
        with c2:
            df_fin = pd.DataFrame({
                "Item": ["Total Revenue", "Net Income", "Total Debt", "Cash"],
                "Value": [
                    fmt_financial_value(fin_data.get('Revenue')),
                    fmt_financial_value(fin_data.get('Net Income')),
                    fmt_financial_value(fin_data.get('Total Debt')),
                    fmt_financial_value(fin_data.get('Cash'))
                ]
            })
            st.table(df_fin)
            if not is_real:
                st.caption("⚠️ Network/API Error - Showing Simulated Data")

    with tab2:
        sv = shap_values[idx]
        feature_names: List[str] = [f.replace('industry_', 'Sector: ') for f in sv.feature_names]
        df_shap = pd.DataFrame({'Feature': feature_names, 'SHAP': sv.values})
        df_shap = df_shap.sort_values('SHAP', ascending=True).tail(6)
        colors: List[str] = ['#d32f2f' if x > 0 else '#388e3c' for x in df_shap['SHAP']]
        shap_fig = go.Figure(go.Bar(
            x=df_shap['SHAP'],
            y=df_shap['Feature'],
            orientation='h',
            marker_color=colors
        ))
        shap_fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Risk Contribution (Log-odds)")
        st.plotly_chart(shap_fig, use_container_width=True)

    # Credit Memo
    st.markdown("---")
    st.markdown("### 📝 Internal Credit Memorandum")

    report_html: str = generate_risk_report_for_company(selected_company, X_test_df, shap_values)

    recommendation_html: str = (
        '<span style="color:red; font-weight:bold; border:1px solid red; padding:3px;">REJECT / WATCHLIST</span>'
        if new_pd > 0.2
        else '<span style="color:green; font-weight:bold; border:1px solid green; padding:3px;">APPROVE</span>'
    )

    st.markdown(f"""
    <div class="report-container">
        <div style="border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 15px;">
            <h3>CREDIT RISK ASSESSMENT: {name}</h3>
            <span><strong>DATE:</strong> {datetime.today().strftime('%Y-%m-%d')}</span> &nbsp;|&nbsp;
            <span><strong>SOURCE:</strong> {'HKEX API' if is_real else 'Internal Sim'}</span>
        </div>
        {report_html}
        <br>
        <div style="background-color: #fff3e0; padding: 15px; border: 1px solid #ffe0b2; margin-top: 15px;">
            <strong>📉 STRESS TEST ADDENDUM:</strong><br>
            Under simulated macro-economic stress (Revenue {shock_rev:+.0%}, Volatility {shock_vol:.1f}x),
            the Probability of Default (PD) shifts from <strong>{base_pd:.1%}</strong> to <strong>{new_pd:.1%}</strong>.
            <br><br>
            <strong>FINAL RECOMMENDATION:</strong> {recommendation_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
