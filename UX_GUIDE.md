# Credit One - 用户体验文档

## 📋 项目概述
**Credit One** 是一个生产级信用风险引擎，实现 Basel III / IFRS 9 合规的 PD（违约概率）预测，包含完整的模型风险管理框架。

---

## 🚀 3分钟上手

### 步骤1: Clone & Install
```bash
git clone <repo-url> credit-one
cd credit-one
pip install -r requirements.txt
```

**依赖清单**:
- pandas>=1.3.0, numpy>=1.21.0
- scikit-learn>=1.0.0, xgboost>=1.7.0
- shap>=0.41.0 (模型可解释性)
- streamlit>=1.30.0 (可视化界面)

### 步骤2: 运行第一个输出
```bash
# 方式1: 运行模型验证套件（推荐）
python model_validation.py

# 方式2: 启动交互式界面
streamlit run app.py
```

**预期输出**:
```
============================================================
MODEL VALIDATION REPORT: XGBoost_PD_Model v2.0_PRODUCTION
============================================================
[1/4] Out-of-Time Validation
  Train AUC: 0.8934
  OOT AUC: 0.8712
  Degradation: 0.0222 ✓
  PSI: 0.0891 ✓
...
```

---

## 🎯 10分钟跑通

### 核心功能理解

| 模块 | 功能 | 运行命令 |
|------|------|----------|
| `model_validation.py` | SR 11-7 合规验证 | `python model_validation.py` |
| `psi_monitoring.py` | PSI 稳定性监控 | `python psi_monitoring.py` |
| `app.py` | Streamlit 风险终端 | `streamlit run app.py` |
| `sme_credit_explainability.py` | SHAP 可解释性 | 被 app.py 调用 |

### 完整运行流程

```bash
# 1. 模型验证（生成合规报告）
python model_validation.py

# 2. PSI 监控模拟
python psi_monitoring.py

# 3. 启动交互式风险评估界面
streamlit run app.py
```

**界面功能**:
- 实时市场数据展示 (腾讯、汇丰、友邦等港股)
- SHAP 特征重要性解释
- 压力测试场景模拟
- 风险报告 PDF 导出

---

## 📊 30分钟接入真实数据

### 配置说明

#### 1. 数据源配置
项目支持以下真实数据源:
- **百行征信** (PBOC Credit Bureau)
- **央行征信** (Central Bank Credit)
- **运营商数据** (Telco Data)
- **电商数据** (E-commerce Data)

#### 2. 数据映射配置
编辑 `transform_logic.sql`:
```sql
-- 示例: 百行征信字段映射
SELECT 
    customer_id,
    credit_score as pboc_score,
    overdue_count_12m,
    total_credit_limit
FROM pboc_credit_report
```

#### 3. 模型配置
编辑 `sme_credit_explainability.py` 中的 `SMEConfig`:
```python
class SMEConfig:
    N_SAMPLES: int = 5000  # 真实数据时替换为实际样本数
    RANDOM_STATE: int = 42
```

### 真实数据运行步骤

```bash
# 1. 准备数据文件
cp your_data.csv data/sme_credit_data.csv

# 2. 修改数据加载逻辑
# 编辑 sme_credit_explainability.py 中的 generate_synthetic_sme_data()

# 3. 重新运行
python model_validation.py
streamlit run app.py
```

---

## ❓ FAQ (5个最常见问题)

### Q1: 运行 `model_validation.py` 报错 "No module named 'xgboost'"
**A**: 安装 XGBoost 依赖:
```bash
pip install xgboost>=1.7.0
# 或
pip install -r requirements.txt
```

### Q2: Streamlit 界面无法加载实时股价
**A**: 检查网络连接，或修改 `REAL_TICKER_MAP` 使用本地数据:
```python
# app.py 中注释掉实时数据获取
# stock = yf.Ticker(ticker)
```

### Q3: SHAP 解释图显示为空白
**A**: 确保已安装 shap 并重启 Streamlit:
```bash
pip install shap>=0.41.0
streamlit run app.py
```

### Q4: 如何添加新的风险特征?
**A**: 修改 `BUSINESS_INSIGHTS` 字典和 `generate_synthetic_sme_data()`:
```python
BUSINESS_INSIGHTS["new_feature"] = {
    "name": "New Feature",
    "threshold": 0.5,
    "why_risk": "...",
    "why_safe": "...",
}
```

### Q5: 模型验证报告中的 AUC 阈值是多少?
**A**: 
- AUC Degradation < 0.05 (可接受)
- PSI Score < 0.25 (稳定)
- ECE < 0.05 (校准良好)

---

## 🚧 上手阻断点清单

### P0 (阻断性)
| 问题 | 影响 | 解决方案 |
|------|------|----------|
| XGBoost 安装失败 | 无法运行模型 | `pip install xgboost --no-binary :all:` |
| Python < 3.9 | 依赖不兼容 | 升级 Python 到 3.9+ |

### P1 (高优先级)
| 问题 | 影响 | 解决方案 |
|------|------|----------|
| SHAP 可视化失败 | 无法解释模型 | 安装 `matplotlib` 和 `shap` |
| Streamlit 端口占用 | 无法启动界面 | `streamlit run app.py --server.port 8502` |
| 实时股价 API 限制 | 市场数据缺失 | 使用本地缓存数据 |

### P2 (中优先级)
| 问题 | 影响 | 解决方案 |
|------|------|----------|
| 合成数据与真实数据差异大 | 模型效果不佳 | 调整 `generate_synthetic_sme_data()` 参数 |
| 报告中文乱码 | PDF 导出异常 | 安装中文字体 `brew install font-wqy-zenhei` |

---

## 📸 截图计划

| 截图位置 | 描述 | 优先级 |
|----------|------|--------|
| `app.py` 主界面 | 风险仪表盘概览 | P0 |
| SHAP 解释图 | 特征重要性瀑布图 | P0 |
| 模型验证报告 | AUC/PSI 指标展示 | P1 |
| 压力测试界面 | 场景模拟结果 | P1 |
| 实时股价面板 | 港股行情展示 | P2 |

---

## 🔗 相关文档

- [scorecard_calibration.md](scorecard_calibration.md) - FICO 评分校准
- [data_architecture.md](data_architecture.md) - 生产数据架构
- [model_governance.md](model_governance.md) - 模型风险管理框架
