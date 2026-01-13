# Algorithmic Credit Risk Engine

Enterprise-grade credit risk management system with IFRS 9 ECL pipeline and real-time SME default prediction. Basel III compliant with live market data integration and interactive stress testing.

## Overview

This repository implements two complementary credit risk modeling systems:

1. **IFRS 9 ECL Pipeline** - Database-centric portfolio risk management with SQL-based feature engineering
2. **SME Credit Default Prediction** - Real-time individual entity risk assessment with XGBoost and SHAP explainability

## Architecture

### IFRS 9 ECL Pipeline

```
Raw Data → SQL ETL → SQL Feature Engineering → Python Modeling → SQL Post-Processing → Reporting
```

**Components:**
- SQL-based ETL and feature engineering (DTI ratio, FICO bucketing)
- Python statistical modeling (Logistic Regression for PD estimation)
- IFRS 9 staging logic (Stage 1/2/3) and ECL calculation
- Database-centric architecture for scalability and auditability

**Usage:**
```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Run with synthetic data
python main.py

# Run with real Lending Club data
python main.py --real-data --samples 50000
```

### SME Credit Risk Engine

**Technology Stack:**
- Modeling: XGBoost with imbalanced class handling
- Explainability: SHAP (Shapley Additive exPlanations)
- Frontend: Streamlit interactive dashboard
- Data Source: Yahoo Finance API (HKEX) with circuit breaker fallback

**System Flow:**
```
Synthetic Data → XGBoost Training → SHAP Attribution → Live Market Data → Streamlit Dashboard
```

**Features:**
- Real-time HKEX market data integration (6-month historical data)
- Circuit breaker pattern for high availability (automatic fallback to simulated data)
- Interactive stress testing (revenue shock simulation, volatility multiplier)
- SHAP-based risk attribution with business-context explanations
- Data caching (1-hour TTL, 95% API call reduction)

**Usage:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive dashboard
streamlit run app.py
```

The dashboard will generate synthetic SME dataset (if not exists), train XGBoost model with SHAP explainability, and launch at `http://localhost:8501`.

**Demo Entities (HKEX Tickers):**
- `HK_00000` → `700.HK` (Tencent Holdings)
- `HK_00001` → `5.HK` (HSBC Holdings)
- `HK_00002` → `1299.HK` (AIA Group)
- `HK_00003` → `3690.HK` (Meituan)
- `HK_00004` → `9988.HK` (Alibaba Group)
- `HK_00005` → `388.HK` (Hong Kong Exchanges)

## Project Structure

```
.
├── app.py                          # Streamlit dashboard
├── sme_credit_explainability.py    # Backend (modeling + SHAP)
├── main.py                         # IFRS 9 ECL pipeline
├── pipeline.py                     # ELT version
├── transform_logic.sql             # SQL feature engineering
├── schema.sql                      # Database schema
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Technical Implementation

### Production-Grade Features

**High Availability:**
- Circuit breaker pattern for API failures
- Automatic fallback to simulated data
- Zero-downtime degradation

**Performance Optimization:**
- Streamlit caching (1-hour TTL)
- Exponential backoff retry logic
- Efficient SHAP computation

**Error Handling:**
- Graceful error messages
- Comprehensive logging
- Data validation at each layer

### Model Performance

**IFRS 9 ECL Pipeline:**
- Model: Logistic Regression
- Evaluation: AUC, Classification Report, Confusion Matrix
- Output: PD predictions, IFRS 9 staging, ECL calculations

**SME Credit Risk Engine:**
- Model: XGBoost (with `scale_pos_weight` for imbalanced data)
- Evaluation: AUC-ROC, Precision/Recall/F1
- Explainability: SHAP TreeExplainer
- Output: Individual entity PD, risk factor attribution, credit memos

## Requirements

See `requirements.txt` for full dependency list. Core dependencies:

- pandas
- numpy
- scikit-learn
- xgboost
- shap
- streamlit
- yfinance

## License

This project is for educational and portfolio demonstration purposes.

## Acknowledgments

- IFRS 9 Standard for ECL framework
- Basel III for credit risk guidelines
- SHAP for model explainability
- Yahoo Finance for market data API
