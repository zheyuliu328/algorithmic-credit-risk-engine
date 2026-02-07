# Algorithmic Credit Risk Engine

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**面向风险建模与研究的信用风险评分工具，实现 PD 预测与模型验证框架。**

---

## 核心能力

1. **双模型架构**: XGBoost（性能）+ Scorecard（可解释性）的混合预测方案
2. **模型验证框架**: Out-of-Time 验证、K-S 检验、PSI 稳定性监控
3. **可解释性**: SHAP 特征归因，支持单样本解释

---

## Quickstart (3 分钟)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行模型验证
python model_validation.py

# 3. 启动交互式界面
streamlit run app.py
```

**输出工件**:
- 模型验证报告（AUC、K-S、PSI 指标）
- 交互式风险评估界面
- SHAP 解释可视化

---

## 模型验证结果

| 指标 | 训练集 | OOT | 状态 |
|:-----|:-------|:----|:-----|
| AUC | 0.8934 | 0.8712 | ✓ |
| PSI | - | 0.0891 | 稳定 |
| K-S | 0.5234 | - | 强区分能力 |

---

## 文档导航

| 文档 | 内容 | 阅读时间 |
|:-----|:-----|:---------|
| [docs/quickstart.md](docs/quickstart.md) | 详细快速入门、预期输出验证 | 10 分钟 |
| [docs/configuration.md](docs/configuration.md) | 数据接入配置、字段映射 | 30 分钟 |
| [docs/faq.md](docs/faq.md) | 常见问题与故障排查 | 按需查阅 |
| [scorecard_calibration.md](scorecard_calibration.md) | FICO 评分校准 | 参考 |
| [data_architecture.md](data_architecture.md) | 数据架构设计 | 参考 |
| [model_governance.md](model_governance.md) | 模型风险管理框架 | 参考 |

---

## 项目结构

```
credit-one/
├── docs/                      # 用户文档
│   ├── quickstart.md         # 10 分钟跑通指南
│   ├── configuration.md      # 30 分钟接入配置
│   └── faq.md                # 常见问题
├── config/
│   └── config.yaml           # 配置文件
├── app.py                    # Streamlit 交互界面
├── model_validation.py       # 模型验证套件
├── psi_monitoring.py         # PSI 监控
├── sme_credit_explainability.py  # 风险引擎核心
└── README.md                 # 本文件
```

---

## 技术栈

| 组件 | 技术 | 用途 |
|:-----|:-----|:-----|
| 模型 | XGBoost, Scorecard | PD 预测 |
| 验证 | scikit-learn, scipy | 统计检验 |
| 解释性 | SHAP | 特征归因 |
| 界面 | Streamlit | 交互可视化 |

---

## 作者

**Zheyu Liu** - 面向风险建模、审计与研究的工具开发

---

*面向风险建模、审计与研究的工具*
