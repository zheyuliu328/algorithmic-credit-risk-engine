# Algorithmic Credit Risk Engine

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**面向信用风险建模研究的违约概率预测工具，演示 XGBoost 评分卡、SHAP 可解释性与模型验证方法。**

---

## 一句话定位

面向信用风险建模研究的违约概率预测工具，演示 XGBoost 评分卡、SHAP 可解释性与模型验证方法。

---

## 核心能力

1. **双模型架构**: XGBoost 非线性模型 + 逻辑回归评分卡，平衡性能与可解释性
2. **模型验证框架**: OOT 验证、K-S 检验、PSI 监控、ECE 校准评估
3. **SHAP 可解释性**: 分解单样本预测的风险因子贡献，生成业务友好的解释

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行模型验证
python model_validation.py

# 3. 启动交互演示
streamlit run app.py
```

### 输出工件

运行后生成：
- `artifacts/demo_report.json` - 模型性能指标（AUC、K-S、Gini）
- 控制台验证报告 - OOT 验证、PSI 稳定性、校准评估
- Streamlit 界面 - 单样本预测与 SHAP 解释可视化

---

## 模型验证结果

### Out-of-Time 验证

| 指标 | 训练集 | OOT | 衰减 | 状态 |
|:-----|:-------|:----|:-----|:-----|
| AUC | 0.8934 | 0.8712 | 0.0222 | ✅ 通过 |
| PSI | - | 0.0891 | - | ✅ 稳定 |

### 区分能力

| 检验 | 分值 | 解释 |
|:-----|:-----|:-----|
| K-S | 0.5234 | 强 |
| Gini | 0.7423 | 优秀 |
| AR | 0.6845 | 良好 |

### 校准

| 指标 | 数值 | 阈值 | 状态 |
|:-----|:-----|:-----|:-----|
| ECE | 0.0234 | < 0.05 | ✅ 通过 |

---

## 项目结构

```
credit-one/
├── app.py                          # Streamlit 交互界面
├── main.py                         # IFRS 9 流程演示
├── pipeline.py                     # 数据管道类
├── sme_credit_explainability.py   # 风险引擎核心
├── model_validation.py            # 模型验证脚本
├── psi_monitoring.py              # PSI 监控演示
├── scorecard_calibration.md       # 评分卡校准文档
├── data_architecture.md           # 数据架构设计
├── model_governance.md            # 模型治理框架
├── INTERVIEW_GUIDE.md             # 面试指南
├── artifacts/
│   └── demo_report.json           # 演示输出
├── docs/
│   ├── glossary.md                # 术语表
│   └── limitations.md             # 限制说明
├── transform_logic.sql            # SQL 特征工程
├── schema.sql                     # 数据库架构
└── requirements.txt               # 依赖
```

---

## 文档索引

| 文档 | 说明 |
|:-----|:-----|
| [docs/glossary.md](docs/glossary.md) | 术语表（PD、K-S、ECE、SHAP 等） |
| [docs/limitations.md](docs/limitations.md) | 项目限制与使用边界 |
| [scorecard_calibration.md](scorecard_calibration.md) | PDO 推导与分数映射 |
| [data_architecture.md](data_architecture.md) | 数据集成架构设计 |
| [model_governance.md](model_governance.md) | 模型治理框架（参考 SR 11-7） |
| [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) | 面试准备指南 |

---

## 参考框架说明

本项目设计参考以下监管框架思路，**非合规认证声明**：

| 框架 | 关系 | 说明 |
|:-----|:-----|:-----|
| Basel III | 对齐思路 | 设计参考 PD 估计方法，非合规认证 |
| IFRS 9 | 支持逻辑 | 演示 ECL 计算流程，非合规工具 |
| SR 11-7 | 参考框架 | 体现 MRM 思维，非合规声明 |

---

## 项目定位与限制

### 项目性质

**本项目是面向信用风险建模研究的教育演示工具**，用于展示评分卡构建、模型验证与可解释性方法。

### 明确限制

| 限制项 | 说明 |
|:-------|:-----|
| ❌ 非合规系统 | 未获得任何监管合规认证（非 Basel III/IFRS 9/SR 11-7 合规系统） |
| ❌ 合成数据 | 使用合成数据训练，模型未在真实信贷数据上验证 |
| ❌ 架构设计 | 百行/央行征信连接为架构设计，未实际实现 |
| ❌ 概念文档 | 模型治理框架为概念文档，未经过实际审计验证 |

### 适用场景

- ✅ 风控/建模岗位面试项目演示
- ✅ 信用评分模型方法论学习
- ✅ 模型验证流程参考

### 生产使用需补充

- 真实信贷数据获取与脱敏处理
- 合规审查与法律评估
- 独立模型验证团队审计
- 生产级监控与预警系统

---

## 技术栈

| 组件 | 用途 |
|:-----|:-----|
| Python 3.9+ | 核心实现 |
| XGBoost | 梯度提升模型 |
| SHAP | 模型可解释性 |
| scikit-learn | 评分卡与验证指标 |
| Streamlit | 交互演示界面 |
| SQLite | 数据存储 |

---

## 作者

**Zheyu Liu**

- LinkedIn: [linkedin.com/in/zheyu-liu-nero0328](https://linkedin.com/in/zheyu-liu-nero0328)
- Email: zheyuliu328@gmail.com

---

**面向风险建模研究 • 演示级实现 • 非生产系统**
