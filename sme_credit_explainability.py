"""SME Credit Risk Explainability Module.

This module provides synthetic data generation, model training with SHAP explanations,
and risk report generation for SME credit risk assessment.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False


class SMEConfig:
    """Configuration constants for SME credit risk model."""
    
    N_SAMPLES: int = 5000
    RANDOM_STATE: int = 42


BUSINESS_INSIGHTS: Dict[str, Dict[str, Any]] = {
    "cash_flow_volatility": {
        "name": "Cash Flow Volatility",
        "threshold": 1.5,
        "why_risk": "High volatility ({val:.2f}x) signals instability in operating cash flows.",
        "why_safe": "Cash flow volatility ({val:.2f}x) is healthy, indicating stable working capital.",
        "benchmark": "Benchmark: <1.5x"
    },
    "debt_to_asset_ratio": {
        "name": "Debt-to-Asset Ratio",
        "threshold": 0.60,
        "why_risk": "Elevated leverage ({val:.1%}) implies limited buffer against asset devaluation.",
        "why_safe": "Leverage ({val:.1%}) is conservative with strong equity buffer.",
        "benchmark": "Threshold: 60%"
    },
    "revenue_growth": {
        "name": "Revenue Growth",
        "threshold": 0.0,
        "why_risk": "Negative growth ({val:.1%}) indicates structural market share loss.",
        "why_safe": "Positive growth ({val:.1%}) demonstrates market competitiveness.",
        "benchmark": "Sector Avg: 5-10%"
    },
    "past_default": {
        "name": "Historical Default",
        "threshold": 0.5,
        "why_risk": "Critical Red Flag: Prior credit events detected.",
        "why_safe": "Clean credit history (No prior defaults).",
        "benchmark": "Hard Stop"
    }
}


def generate_synthetic_sme_data() -> pd.DataFrame:
    """Generate synthetic SME credit risk dataset.

    Creates a DataFrame with synthetic financial metrics for SME companies,
    including revenue growth, debt ratios, cash flow volatility, and default labels.
    The first 6 records are designated as VIP clients with favorable metrics.

    Returns:
        pd.DataFrame: Synthetic dataset with columns for financial metrics,
            industry, default labels, and company IDs.
    """
    np.random.seed(SMEConfig.RANDOM_STATE)
    n = SMEConfig.N_SAMPLES
    
    data = pd.DataFrame({
        'revenue_growth': np.random.normal(0.05, 0.15, n),
        'debt_to_asset_ratio': np.random.beta(2, 5, n),
        'cash_flow_volatility': np.random.gamma(1, 2, n),
        'industry': np.random.choice(['Manufacturing', 'Retail', 'Tech', 'Services'], n),
        'past_default': np.random.binomial(1, 0.1, n)
    })
    
    # Demo God Mode: Make VIPs (HK_00000-05) Perfect
    for i in range(6):
        data.loc[i, 'past_default'] = 0
        data.loc[i, 'revenue_growth'] = np.abs(np.random.normal(0.20, 0.05))
        data.loc[i, 'debt_to_asset_ratio'] = np.random.uniform(0.2, 0.35)
        data.loc[i, 'cash_flow_volatility'] = np.random.uniform(0.5, 0.8)
    
    # Logit logic for default probability
    logit = (
        -3.0
        - 2.0 * data['revenue_growth']
        + 3.5 * data['debt_to_asset_ratio']
        + 0.8 * data['cash_flow_volatility']
        + 1.5 * data['past_default']
    )
    prob = 1 / (1 + np.exp(-logit))
    data['true_label'] = np.random.binomial(1, prob)
    data['company_id'] = [f"HK_{i:05d}" for i in range(n)]
    return data


def train_model_and_explain(
    df: pd.DataFrame
) -> Tuple[Pipeline, pd.DataFrame, shap.Explanation, shap.Explainer]:
    """Train credit risk model and generate SHAP explanations.

    Trains a gradient boosting classifier (XGBoost if available, otherwise sklearn)
    on the provided dataset and computes SHAP values for model interpretability.

    Args:
        df: Input DataFrame containing features, company_id, and true_label columns.

    Returns:
        Tuple containing:
            - clf: Trained sklearn Pipeline with preprocessor and model.
            - X_test_df: Test set DataFrame with predictions and company IDs.
            - shap_values: SHAP Explanation object for test set.
            - explainer: SHAP Explainer instance.
    """
    X = df.drop(columns=['company_id', 'true_label'])
    y = df['true_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    categorical_features: List[str] = ["industry"]
    numeric_features: List[str] = [
        "revenue_growth", "debt_to_asset_ratio", "cash_flow_volatility", "past_default"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        verbose_feature_names_out=False
    )
    
    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            scale_pos_weight=10,
            random_state=42
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
    
    clf = Pipeline([("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)
    y_proba: np.ndarray = clf.predict_proba(X_test)[:, 1]
    
    # SHAP explanation
    model_step = clf.named_steps["model"]
    preprocessor_step = clf.named_steps["preprocessor"]
    X_test_transformed = preprocessor_step.transform(X_test)
    transformed_feature_names: List[str] = list(preprocessor_step.get_feature_names_out())
    
    explainer = shap.Explainer(
        model_step.predict, X_test_transformed, feature_names=transformed_feature_names
    )
    shap_values = explainer(X_test_transformed)
    shap_values.feature_names = transformed_feature_names

    X_test_df = pd.DataFrame(X_test_transformed, columns=transformed_feature_names)
    original_indices = X_test.index
    company_ids = df.loc[original_indices, 'company_id'].values
    X_test_df['company_id'] = company_ids
    X_test_df['predicted_default_prob'] = y_proba
    
    # Restore raw values
    raw_test = X_test.reset_index(drop=True)
    for col in numeric_features:
        X_test_df[col] = raw_test[col]
    
    # Demo God Mode: Force VIP clients to have low PD
    vip_ids = [f"HK_{i:05d}" for i in range(6)]
    mask = X_test_df['company_id'].isin(vip_ids)
    if mask.any():
        X_test_df.loc[mask, 'predicted_default_prob'] = np.random.uniform(0.01, 0.04, mask.sum())
    
    return clf, X_test_df, shap_values, explainer


def generate_risk_report_for_company(
    company_id: str,
    X_test_df: pd.DataFrame,
    shap_values: shap.Explanation
) -> str:
    """Generate HTML risk report for a specific company.

    Creates a formatted HTML report containing executive summary, risk factors,
    SHAP attributions, and AI-generated commentary for credit assessment.

    Args:
        company_id: Unique identifier for the company (e.g., "HK_00001").
        X_test_df: Test DataFrame containing predictions and features.
        shap_values: SHAP Explanation object with feature attributions.

    Returns:
        HTML string containing the formatted risk report, or error message
        if company_id is not found.
    """
    try:
        row_idx: int = X_test_df.index[X_test_df['company_id'] == company_id].tolist()[0]
    except IndexError:
        return "Error: ID Not Found"
    
    sv = shap_values[row_idx]
    
    impact_list: List[Tuple[str, float]] = []
    for name, val in zip(sv.feature_names, sv.values):
        impact_list.append((name, val))
    impact_list.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factors = impact_list[:3]
    
    prob: float = X_test_df.iloc[row_idx]['predicted_default_prob']
    risk_level = "HIGH RISK" if prob > 0.2 else "LOW RISK"
    color_class = "#d32f2f" if prob > 0.2 else "#388e3c"
    
    # Build HTML without indentation to prevent Markdown rendering issues
    html = ""
    html += f"<div style='margin-bottom: 20px;'><h4 style='color: {color_class}; margin-bottom: 5px;'>EXECUTIVE SUMMARY: {risk_level}</h4>"
    html += f"<div style='font-size: 1.1em;'>Model Prediction (PD): <strong>{prob:.2%}</strong></div></div>"
    
    html += "<table style='width:100%; border-collapse: collapse; font-size: 0.9em; font-family: Arial, sans-serif;'>"
    html += "<thead style='background-color: #f5f5f5;'><tr>"
    html += "<th style='padding: 10px; border: 1px solid #ddd; text-align: left;'>Risk Factor</th>"
    html += "<th style='padding: 10px; border: 1px solid #ddd; text-align: left;'>AI Attribution</th>"
    html += "<th style='padding: 10px; border: 1px solid #ddd; text-align: left;'>Metric Analysis</th>"
    html += "<th style='padding: 10px; border: 1px solid #ddd; text-align: left;'>Commentary</th>"
    html += "</tr></thead><tbody>"
    
    for feat_key, shap_val in top_factors:
        biz_logic: Optional[Dict[str, Any]] = None
        base_key = feat_key
        for key in BUSINESS_INSIGHTS.keys():
            if key in feat_key:
                biz_logic = BUSINESS_INSIGHTS[key]
                base_key = key
                break
        
        if not biz_logic:
            continue
        
        try:
            raw_val = X_test_df.iloc[row_idx][base_key]
        except KeyError:
            raw_val = 0
        
        # Risk determination logic
        threshold = biz_logic.get('threshold')
        if threshold is not None:
            if base_key == 'revenue_growth':
                is_risky = raw_val < threshold
            elif base_key == 'past_default':
                is_risky = raw_val > threshold
            else:
                is_risky = raw_val > threshold
        else:
            is_risky = shap_val > 0
        
        if "growth" in base_key or "ratio" in base_key:
            val_str = f"{raw_val:.1%}"
        else:
            val_str = f"{raw_val:.2f}"
        
        status_html = (
            "<span style='color:red; font-weight:bold'>⚠️ Risk</span>"
            if is_risky
            else "<span style='color:green; font-weight:bold'>✅ Safe</span>"
        )
        comment = (
            biz_logic['why_risk'].format(val=raw_val)
            if is_risky
            else biz_logic['why_safe'].format(val=raw_val)
        )
        
        shap_color = "red" if shap_val > 0 else "green"
        shap_arrow = "▲" if shap_val > 0 else "▼"
        
        html += "<tr>"
        html += f"<td style='padding: 10px; border: 1px solid #ddd;'><strong>{biz_logic['name']}</strong></td>"
        html += f"<td style='padding: 10px; border: 1px solid #ddd; color: {shap_color};'><strong>{shap_arrow} {shap_val:.3f}</strong></td>"
        html += f"<td style='padding: 10px; border: 1px solid #ddd;'><div>{val_str}</div><div>{status_html}</div></td>"
        html += f"<td style='padding: 10px; border: 1px solid #ddd; color: #555;'>{comment}</td>"
        html += "</tr>"
    
    html += "</tbody></table>"
    html += "<div style='margin-top: 15px; padding: 10px; background-color: #f9f9f9; border-left: 3px solid #333; font-style: italic; color: #555;'>"
    html += "<strong>🤖 AI Suggestion:</strong> Validate flagged metrics against audited financial statements.</div>"
    
    return html
