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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_extras.metric_cards import style_metric_cards

from sme_credit_explainability import (
    generate_synthetic_sme_data,
    train_model_and_explain,
    train_scorecard_model,
    generate_risk_report_for_company,
    predict_from_live_data,
    calculate_psi,
    monitor_model_stability,
    BUSINESS_INSIGHTS,
    HAS_OPTBINNING,
)

# Page Config
st.set_page_config(page_title="CreditOne | HK Risk Terminal", layout="wide", page_icon="🏙️")

def init_ui_style():
    """Initialize Dark Knight terminal-style UI."""
    st.markdown("""
        <style>
        /* Remove top padding */
        .block-container {
            padding-top: 3.5rem;
            padding-bottom: 0rem;
        }
        /* Terminal-like font */
        html, body, [class*="css"] {
            font-family: 'Roboto Mono', monospace; 
        }
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #111;
        }
        /* Enhanced metric cards */
        div[data-testid="metric-container"] {
            background-color: #222;
            border: 1px solid #333;
            padding: 10px;
            border-radius: 5px;
            color: #eee;
        }
        /* Metric value color */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        .insight-box-risk { background-color: #ffebee; border-left: 5px solid #d32f2f; padding: 15px; border-radius: 4px; color: #b71c1c; }
        .insight-box-safe { background-color: #e8f5e9; border-left: 5px solid #2e7d32; padding: 15px; border-radius: 4px; color: #1b5e20; }
        .report-container { background-color: white; padding: 30px; border: 1px solid #ccc; font-family: 'Times New Roman', serif; margin-top: 20px; }
        .section-header {
            color: #88c0d0;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 2px solid #2e3440;
        }
        </style>
    """, unsafe_allow_html=True)

def display_pro_table(df, height=300):
    """Display professional interactive table with AgGrid."""
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    AgGrid(
        df, 
        gridOptions=gridOptions,
        enable_enterprise_modules=False,
        height=height, 
        theme='balham',
        update_mode=GridUpdateMode.SELECTION_CHANGED
    )

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
    """Format financial value for display."""
    if v is None or v == 0:
        return "N/A"
    return f"{v/1e9:.2f} B" if v > 1e9 else f"{v:,.0f}"


def _generate_live_report(metrics: Dict[str, Any], shap_vals) -> str:
    """Generate HTML report from live metrics and SHAP values."""
    rev = metrics.get('revenue_growth', 0)
    debt = metrics.get('debt_to_asset_ratio', 0)
    vol = metrics.get('cash_flow_volatility', 0)
    
    html = "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
    html += "<thead style='background:#f5f5f5;'><tr>"
    html += "<th style='padding:10px;border:1px solid #ddd;'>Risk Factor</th>"
    html += "<th style='padding:10px;border:1px solid #ddd;'>Value</th>"
    html += "<th style='padding:10px;border:1px solid #ddd;'>Status</th>"
    html += "</tr></thead><tbody>"
    
    # Revenue Growth
    rev_status = "✅ Safe" if rev >= 0 else "⚠️ Risk"
    html += f"<tr><td style='padding:10px;border:1px solid #ddd;'>Revenue Growth</td>"
    html += f"<td style='padding:10px;border:1px solid #ddd;'>{rev:.1%}</td>"
    html += f"<td style='padding:10px;border:1px solid #ddd;'>{rev_status}</td></tr>"
    
    # Debt-to-Asset
    debt_status = "⚠️ Risk" if debt > 0.6 else "✅ Safe"
    html += f"<tr><td style='padding:10px;border:1px solid #ddd;'>Debt-to-Asset Ratio</td>"
    html += f"<td style='padding:10px;border:1px solid #ddd;'>{debt:.1%}</td>"
    html += f"<td style='padding:10px;border:1px solid #ddd;'>{debt_status}</td></tr>"
    
    # Cash Flow Volatility
    vol_status = "⚠️ Risk" if vol > 1.5 else "✅ Safe"
    html += f"<tr><td style='padding:10px;border:1px solid #ddd;'>Cash Flow Volatility</td>"
    html += f"<td style='padding:10px;border:1px solid #ddd;'>{vol:.2f}x</td>"
    html += f"<td style='padding:10px;border:1px solid #ddd;'>{vol_status}</td></tr>"
    html += "</tbody></table>"
    return html


def main() -> None:
    """Main entry point for the Streamlit dashboard.

    Initializes the risk engine, renders UI components, and handles
    user interactions for stress testing and company selection.
    """
    # Initialize Dark Knight UI
    init_ui_style()
    
    # Header
    st.markdown("### 🏦 CreditOne V6.0 | Terminal")
    
    # Init Data & Model
    if 'model_ready' not in st.session_state:
        with st.spinner('Initializing Risk Engine...'):
            df = generate_synthetic_sme_data()
            clf, X_test_df, shap_values, explainer = train_model_and_explain(df)
            
            # Train scorecard model if OptBinning available
            scorecard = None
            if HAS_OPTBINNING:
                try:
                    scorecard, _ = train_scorecard_model(df)
                except Exception:
                    pass
            
            st.session_state.update({
                'clf': clf,
                'explainer': explainer,
                'scorecard': scorecard,
                'train_data': df,
                'model_ready': True
            })

    clf = st.session_state['clf']
    explainer = st.session_state['explainer']
    scorecard = st.session_state.get('scorecard')
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Analysis", "📈 Monitoring"])
    
    with tab1:
        render_analysis_tab(clf, explainer, scorecard)
    
    with tab2:
        render_monitoring_tab()


def render_analysis_tab(clf, explainer, scorecard):
    """Render the main analysis tab."""
    # Create three-column grid layout
    left_col, center_col, right_col = st.columns([1, 2.5, 1.5])

    # ==================== LEFT COLUMN: Controls & Key Metrics ====================
    with left_col:
        st.markdown('<div class="section-header">🎛️ CONTROL PANEL</div>', unsafe_allow_html=True)
        
        # Model mode toggle
        if scorecard is not None:
            model_mode = st.radio(
                "Model Mode",
                options=["Agile (XGBoost)", "Compliant (Scorecard)"],
                index=0,
                help="Agile: ML-based, Compliant: Regulatory scorecard"
            )
            use_scorecard = "Compliant" in model_mode
        else:
            use_scorecard = False
            if HAS_OPTBINNING:
                st.caption("⚠️ Scorecard training failed")
        
        # Universal ticker search
        ticker_input: str = st.text_input("Stock Ticker", value="9988.HK", label_visibility="collapsed", placeholder="e.g., 9988.HK")
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        st.markdown('<div class="section-header">🧪 STRESS LAB</div>', unsafe_allow_html=True)
        shock_rev_int: int = st.slider("Revenue Shock (%)", -50, 20, 0, 5)
        shock_rev: float = shock_rev_int / 100.0
        shock_vol: float = st.slider("Vol Multiplier", 1.0, 3.0, 1.0, 0.1)

    # Initialize live result in session state
    if 'live_result' not in st.session_state:
        st.session_state['live_result'] = None
        st.session_state['live_shap'] = None

    # Run live prediction when button clicked
    if analyze_btn and ticker_input:
        with st.spinner(f'Fetching live data for {ticker_input}...'):
            # Use appropriate model
            if use_scorecard and scorecard is not None:
                result, shap_vals = predict_from_live_data(ticker_input, scorecard, None)
            else:
                result, shap_vals = predict_from_live_data(ticker_input, clf, explainer)
            st.session_state['live_result'] = result
            st.session_state['live_shap'] = shap_vals
            st.session_state['use_scorecard'] = use_scorecard

    result = st.session_state.get('live_result')
    live_shap = st.session_state.get('live_shap')
    use_scorecard = st.session_state.get('use_scorecard', False)

    # Handle errors or no data
    if result is None:
        center_col.info("👆 Enter a ticker and click **Analyze** to start real-time risk assessment.")
        return

    if not result.get('success'):
        center_col.error(f"❌ Error: {result.get('error', 'Failed to fetch data')}")
        center_col.info("Try a valid ticker like `700.HK`, `AAPL`, `TSLA`, `9988.HK`")
        return

    # Extract data from live result
    name = result['company_name']
    metrics = result['metrics']
    fin_data = result['fin_data']
    hist = result['hist']
    base_pd = result['pd_prob']
    is_real = True

    # Build row-like dict for compatibility
    row = {
        'revenue_growth': metrics['revenue_growth'],
        'debt_to_asset_ratio': metrics['debt_to_asset_ratio'],
        'cash_flow_volatility': metrics['cash_flow_volatility'],
        'past_default': metrics['past_default'],
        'predicted_default_prob': base_pd
    }

    # Stress Logic
    base_pd: float = np.clip(row['predicted_default_prob'], 0.001, 0.999)
    logit_change: float = (shock_rev * -5.0) + ((shock_vol - 1.0) * 1.5)
    new_pd: float = 1 / (1 + np.exp(-(np.log(base_pd / (1 - base_pd)) + logit_change)))

    str_rev: float = row['revenue_growth'] * (1 + shock_rev)
    str_vol: float = row['cash_flow_volatility'] * shock_vol

    # ==================== LEFT COLUMN: Key Metrics ====================
    with left_col:
        st.markdown('<div class="section-header">📊 KEY METRICS</div>', unsafe_allow_html=True)
        st.metric("Revenue Growth", f"{str_rev:.1%}", delta=f"{shock_rev:.0%}" if shock_rev != 0 else None)
        st.metric("Debt-to-Asset", f"{row['debt_to_asset_ratio']:.1%}")
        st.metric("CF Volatility", f"{str_vol:.2f}", delta=f"{shock_vol:.1f}x" if shock_vol != 1.0 else None, delta_color="inverse")
    
    # ==================== CENTER COLUMN: Market & Model Core ====================
    with center_col:
        # Display credit score for scorecard mode, PD for agile mode
        if use_scorecard and scorecard is not None:
            st.markdown('<div class="section-header">💳 CREDIT SCORE</div>', unsafe_allow_html=True)
            
            # Convert PD to credit score using PDO formula
            def pd_to_score(pd):
                base_score, base_odds, pdo = 600, 50, 20
                factor = pdo / np.log(2)
                offset = base_score - (factor * np.log(base_odds))
                odds = (1 - pd) / np.clip(pd, 1e-10, 0.9999)
                return np.clip(offset + factor * np.log(odds), 300, 850)
            
            base_score = pd_to_score(base_pd)
            new_score = pd_to_score(new_pd)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=new_score,
                title={'text': "Credit Score (FICO-style)"},
                delta={
                    'reference': base_score,
                    'increasing': {'color': "green"},
                    'decreasing': {'color': "red"}
                },
                gauge={
                    'axis': {'range': [300, 850]},
                    'steps': [
                        {'range': [300, 550], 'color': '#ef5350'},
                        {'range': [550, 700], 'color': '#ffa726'},
                        {'range': [700, 850], 'color': '#66bb6a'}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Entity: **{name}** | PD: {new_pd:.2%}")
        else:
            st.markdown('<div class="section-header">🎯 PROBABILITY OF DEFAULT</div>', unsafe_allow_html=True)
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
        
        st.markdown('<div class="section-header">📈 MARKET PRICE TREND</div>', unsafe_allow_html=True)
        price_fig = go.Figure(data=[
            go.Scatter(x=hist.index, y=hist['Close'], line=dict(color='#1976d2', width=2))
        ])
        price_fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(price_fig, use_container_width=True)
        
        st.markdown('<div class="section-header">📋 FINANCIALS</div>', unsafe_allow_html=True)
        df_fin = pd.DataFrame({
            "Item": ["Revenue", "Net Income", "Debt", "Cash"],
            "Value": [
                fmt_financial_value(fin_data.get('Revenue')),
                fmt_financial_value(fin_data.get('Net Income')),
                fmt_financial_value(fin_data.get('Total Debt')),
                fmt_financial_value(fin_data.get('Cash'))
            ]
        })
        display_pro_table(df_fin, height=180)
    
    # ==================== RIGHT COLUMN: Risk Attribution & Insights ====================
    with right_col:
        if result.get('note'):
            st.markdown('<div class="section-header">💎 BLUE-CHIP</div>', unsafe_allow_html=True)
            st.success(result['note'])
        
        st.markdown('<div class="section-header">🔍 RISK DRIVERS</div>', unsafe_allow_html=True)
        
        if use_scorecard and scorecard is not None:
            # Show WoE binning for scorecard mode
            st.caption("**WoE Binning (Regulatory Compliant)**")
            try:
                # Show binning for debt ratio as example
                binning_df = scorecard.get_binning_table('debt_to_asset_ratio')
                st.dataframe(binning_df[['Bin', 'Count', 'Event rate', 'WoE']].head(5), use_container_width=True)
            except Exception:
                st.caption("Binning table unavailable")
        elif live_shap is not None:
            # Show SHAP for XGBoost mode
            sv = live_shap[0]
            feature_names: List[str] = [f.replace('industry_', 'Sector: ') for f in sv.feature_names]
            df_shap = pd.DataFrame({'Feature': feature_names, 'SHAP': sv.values})
            df_shap = df_shap.sort_values('SHAP', key=abs, ascending=False).head(3)
            colors: List[str] = ['#ef5350' if x > 0 else '#66bb6a' for x in df_shap['SHAP']]
            shap_fig = go.Figure(go.Bar(
                x=df_shap['SHAP'],
                y=df_shap['Feature'],
                orientation='h',
                marker_color=colors
            ))
            shap_fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(shap_fig, use_container_width=True)
        
        st.markdown('<div class="section-header">📝 CREDIT MEMO</div>', unsafe_allow_html=True)
        credit_rating = "Investment Grade" if new_pd < 0.15 else "Speculative Grade" if new_pd < 0.30 else "High Risk"
        recommendation = 'APPROVE' if new_pd < 0.20 else 'REVIEW' if new_pd < 0.35 else 'REJECT'
        
        risks: List[str] = []
        if str_rev < 0:
            risks.append(f"Revenue contracting ({str_rev:.1%})")
        if row['debt_to_asset_ratio'] > 0.6:
            risks.append(f"High leverage ({row['debt_to_asset_ratio']:.1%})")
        if str_vol > 1.5:
            risks.append(f"Volatile CF ({str_vol:.2f}x)")
        
        risk_summary = "\n".join([f"• {r}" for r in risks[:3]]) if risks else "• All metrics healthy"
        
        st.info(f"""
**Rating**: {credit_rating}  
**PD**: {new_pd:.2%}  
**Key Risks**:  
{risk_summary}

**Action**: {recommendation}
        """)
        
        st.download_button(
            "📥 Download Report",
            data=f"Credit Report: {name}\nPD: {new_pd:.2%}\nRating: {credit_rating}\nRecommendation: {recommendation}",
            file_name=f"{ticker_input}_report.txt",
            use_container_width=True
        )


def render_monitoring_tab():
    """Render the PSI monitoring tab."""
    st.header("🔍 Model Monitoring Dashboard")
    st.markdown("Monitor data drift and model stability using Population Stability Index (PSI)")
    
    train_data = st.session_state.get('train_data')
    
    if train_data is None:
        st.warning("⚠️ Training data not available. Please restart the application.")
        return
    
    # Data source selection
    st.subheader("📂 Data Source")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Production Data (CSV)", type=['csv'], help="Upload real production data to detect actual drift")
        if uploaded_file:
            try:
                current_data = pd.read_csv(uploaded_file)
                st.session_state['current_data'] = current_data
                st.session_state['data_source'] = 'uploaded'
                st.success(f"✅ Loaded {len(current_data)} records")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    with col2:
        if st.button("🎲 Simulate Drift (Demo)"):
            drifted_data = train_data.copy()
            drifted_data['revenue_growth'] = train_data['revenue_growth'] + np.random.normal(0.02, 0.01, len(train_data))
            drifted_data['debt_to_asset_ratio'] = train_data['debt_to_asset_ratio'] + np.random.normal(0.15, 0.05, len(train_data))
            drifted_data['cash_flow_volatility'] = train_data['cash_flow_volatility'] * 1.5 + np.random.normal(0, 0.2, len(train_data))
            st.session_state['current_data'] = drifted_data
            st.session_state['data_source'] = 'simulated'
            st.success("✅ Drift simulated!")
    
    with col3:
        if st.button("🔄 Reset to Baseline"):
            if 'current_data' in st.session_state:
                del st.session_state['current_data']
                del st.session_state['data_source']
            st.success("✅ Reset to baseline.")
    
    # Get current data and show status
    current_data = st.session_state.get('current_data', train_data)
    data_source = st.session_state.get('data_source', 'baseline')
    
    if data_source == 'baseline':
        st.info("ℹ️ **Baseline Mode**: Comparing training data with itself (PSI = 0). Upload production data or simulate drift to see monitoring in action.")
    elif data_source == 'uploaded':
        st.success(f"✅ **Production Mode**: Monitoring {len(current_data)} production records against {len(train_data)} baseline records.")
    elif data_source == 'simulated':
        st.warning("⚠️ **Demo Mode**: Using simulated drift for demonstration. Upload real data for actual monitoring.")
    
    # Define features to monitor
    feature_cols = ['revenue_growth', 'debt_to_asset_ratio', 'cash_flow_volatility']
    
    # Calculate PSI
    psi_df = monitor_model_stability(train_data, current_data, feature_cols)
    
    # Display PSI table
    st.subheader("📊 PSI Scores by Feature")
    st.dataframe(psi_df, use_container_width=True, hide_index=True)
    
    # Overall alert
    max_psi = psi_df['PSI'].max()
    if max_psi >= 0.25:
        st.error("🚨 **CRITICAL ALERT**: Significant data drift detected! Model retraining is required.")
    elif max_psi >= 0.1:
        st.warning("⚠️ **WARNING**: Moderate data drift detected. Monitor model performance closely.")
    else:
        st.success("✅ **ALL CLEAR**: No significant data drift detected. Model is stable.")
    
    # PSI visualization
    st.subheader("📈 PSI Visualization")
    colors = ['red' if psi >= 0.25 else 'orange' if psi >= 0.1 else 'green' 
              for psi in psi_df['PSI']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=psi_df['Feature'],
        y=psi_df['PSI'],
        marker_color=colors,
        text=psi_df['PSI'].round(4),
        textposition='outside'
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
                  annotation_text="Critical (0.25)", annotation_position="right")
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                  annotation_text="Warning (0.1)", annotation_position="right")
    
    fig.update_layout(
        title="Population Stability Index by Feature",
        xaxis_title="Feature",
        yaxis_title="PSI Value",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # PSI interpretation guide
    with st.expander("ℹ️ How to interpret PSI scores"):
        st.markdown("""
        **Population Stability Index (PSI)** measures the shift in data distribution between two datasets:
        
        - **PSI < 0.1** 🟢: No significant change. Model is stable.
        - **0.1 ≤ PSI < 0.25** 🟡: Moderate change. Monitor model performance.
        - **PSI ≥ 0.25** 🔴: Significant change. Model retraining recommended.
        
        **What causes high PSI?**
        - Market conditions change (e.g., economic crisis)
        - Customer behavior shifts
        - Data collection process changes
        - Seasonal effects
        """)


if __name__ == "__main__":
    main()
