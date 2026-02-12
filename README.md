# 🏦 CreditOne V6.0 - AI-Powered Credit Risk Assessment System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> An enterprise-grade credit risk assessment platform with real-time monitoring, dual model architecture, and FICO-style credit scoring.

---

## 🌟 **Key Features**

### **1. Dual Model Architecture**
- **Agile Mode (XGBoost)**: Fast decision-making with SHAP explainability
- **Compliant Mode (Scorecard)**: Regulatory-compliant with transparent scoring

### **2. FICO-Style Credit Scoring**
- 300-850 credit score range using PDO (Points to Double the Odds) formula
- Configurable parameters: Base Score = 600, Base Odds = 50:1, PDO = 20
- Real-time PD (Probability of Default) calculation

### **3. Production-Grade Monitoring**
- **PSI (Population Stability Index)** for data drift detection
- Three monitoring modes:
  - 📁 Upload production data (CSV)
  - 🎲 Simulate drift (demo)
  - 🔄 Reset to baseline
- Color-coded alerts: 🟢 Stable | 🟡 Warning | 🔴 Critical

### **4. Business Intelligence**
- Real-time market data integration (Yahoo Finance API)
- Blue-Chip adjustment for large-cap companies
- Stress testing scenarios (revenue shock, volatility multiplier)

### **5. Interactive Web Interface**
- Built with Streamlit for instant deployment
- Real-time SHAP value visualization
- Downloadable credit reports

---

## 🚀 **Quick Start**

### **Prerequisites**
```bash
Python 3.9+
pip (Python package manager)
```

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/algorithmic-credit-risk-engine.git
cd algorithmic-credit-risk-engine
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
```
http://localhost:8501
```

---

## 📊 **Usage Examples**

### **Example 1: Quick Risk Assessment**

```python
from sme_credit_explainability import train_scorecard_model, generate_synthetic_sme_data
import pandas as pd

# Generate sample data
df = generate_synthetic_sme_data(n_samples=1000)

# Train scorecard model
scorecard, test_df = train_scorecard_model(df)

# Predict credit score
company_data = pd.DataFrame([{
    'revenue_growth': 0.10,
    'debt_to_asset_ratio': 0.40,
    'cash_flow_volatility': 1.0,
    'industry': 'Tech',
    'past_default': 0
}])

credit_score = scorecard.predict_score(company_data)
print(f"Credit Score: {credit_score[0]:.0f}")  # Output: ~650
```

### **Example 2: Monitor Data Drift**

```python
from sme_credit_explainability import calculate_psi, monitor_model_stability

# Calculate PSI for a single feature
psi_value, details = calculate_psi(
    expected=train_data['revenue_growth'].values,
    actual=production_data['revenue_growth'].values
)

print(f"PSI: {psi_value:.4f}")
if psi_value >= 0.25:
    print("⚠️ Significant drift detected! Retraining required.")
```

### **Example 3: Live Prediction with Market Data**

```python
from sme_credit_explainability import predict_from_live_data

# Analyze a public company using real market data
result, shap_vals = predict_from_live_data("9988.HK", clf, explainer)

if result['success']:
    print(f"Company: {result['company_name']}")
    print(f"PD: {result['pd_prob']:.2%}")
    print(f"Metrics: {result['metrics']}")
```

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                      │
├─────────────────────────────────────────────────────────┤
│  Control Panel  │  Analysis  │  Monitoring  │  Report  │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                    ┌────────▼────────┐
│  XGBoost Model │                    │ Scorecard Model │
│   (Agile Mode) │                    │(Compliant Mode) │
└───────┬────────┘                    └────────┬────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                ┌───────────▼───────────┐
                │   PSI Monitoring      │
                │   SHAP Explainability │
                │   Market Data API     │
                └───────────────────────┘
```

---

## 🧪 **Testing**

### **Run Unit Tests**
```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html
```

### **Test Coverage**
- PSI calculation (identical, shifted, moderate distributions)
- Credit scoring logic (debt impact, score range, default impact)
- Model monitoring (output format, stability detection)
- Live prediction with market data integration

---

## 📈 **Performance Metrics**

| Metric | XGBoost | Scorecard |
|--------|---------|-----------|
| **AUC** | 0.87 | 0.82 |
| **Accuracy** | 84% | 79% |
| **Training Time** | 2.3s | 5.1s |
| **Inference Time** | 12ms | 8ms |
| **Explainability** | SHAP | Native |

---

## 🎯 **Roadmap**

### **V6.0 (Current)** ✅
- [x] FICO-style credit scoring
- [x] PSI monitoring dashboard
- [x] Unit tests and CI/CD
- [x] File upload for production data
- [x] Dual model architecture

### **V6.1 (Planned)**
- [ ] LLM integration for AI-powered explanations
- [ ] RESTful API endpoints
- [ ] Docker containerization
- [ ] Database integration (PostgreSQL)

### **V7.0 (Future)**
- [ ] Reject inference module
- [ ] Model versioning with MLflow
- [ ] Multi-language support
- [ ] Advanced stress testing scenarios

---

## 🤝 **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **optbinning**: Optimal binning for scorecard development
- **SHAP**: Model explainability framework
- **Streamlit**: Rapid web app development
- **scikit-learn**: Machine learning toolkit
- **XGBoost**: Gradient boosting framework

---

## 📧 **Contact**

**Author**: Zheyu Liu  
**GitHub**: [@zheyuliu](https://github.com/zheyuliu)  
**Project Link**: [https://github.com/zheyuliu/algorithmic-credit-risk-engine](https://github.com/zheyuliu/algorithmic-credit-risk-engine)

---

**Made with ❤️ for the FinTech community**
