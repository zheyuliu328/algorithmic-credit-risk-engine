# Quickstart Guide - 10 分钟跑通

> 本指南帮助你在 10 分钟内完整运行 Credit One 系统并验证输出。

---

## 前置要求

- Python 3.9+
- 2GB 可用内存
- 网络连接（用于下载依赖）

---

## 步骤 1: 环境准备 (2 分钟)

```bash
# 克隆项目
git clone <repo-url> credit-one
cd credit-one

# 安装依赖
pip install -r requirements.txt
```

**依赖清单**:
- pandas>=1.3.0, numpy>=1.21.0
- scikit-learn>=1.0.0, xgboost>=1.7.0
- shap>=0.41.0
- streamlit>=1.30.0

**安装验证**:
```bash
python -c "import xgboost; import shap; import streamlit; print('OK')"
```

---

## 步骤 2: 运行模型验证 (3 分钟)

```bash
python model_validation.py
```

**这一步会做什么**:
- ✅ 加载合成数据集
- ✅ 训练 XGBoost 模型
- ✅ 运行 Out-of-Time 验证
- ✅ 计算 K-S、AUC、PSI 指标

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

[2/4] Discrimination Power
  K-S: 0.5234 (Strong)
  Gini: 0.7423 (Excellent)

[3/4] Calibration
  ECE: 0.0234 ✓

[4/4] Summary
  Status: PASS
  Recommendation: Model is stable and ready for deployment
```

---

## 步骤 3: 运行 PSI 监控 (2 分钟)

```bash
python psi_monitoring.py
```

**预期输出**:
```
============================================================
PSI MONITORING SIMULATION
============================================================
[Scenario 1: Stable Period]
🟢 STABLE (PSI: 0.0456)

[Scenario 2: Slight Drift]
🟡 WARNING (PSI: 0.1890)
Alert: Feature drift detected in 2 variables

[Scenario 3: Significant Drift]
🔴 CRITICAL (PSI: 0.3567)
Alert: Model retraining recommended
```

---

## 步骤 4: 启动交互界面 (3 分钟)

```bash
streamlit run app.py
```

**预期看到**:
- 浏览器自动打开 http://localhost:8501
- 风险仪表盘界面
- 实时股价面板（腾讯、汇丰等港股）
- SHAP 解释图表

**界面功能**:
- 查看实时市场数据
- 运行单样本风险评估
- 查看 SHAP 特征重要性
- 导出风险报告

---

## 验证清单

完成上述步骤后，验证以下项目:

- [ ] 模型验证报告生成，AUC > 0.85
- [ ] PSI 监控正常运行，显示三种状态
- [ ] Streamlit 界面可正常访问
- [ ] SHAP 图表正常显示
- [ ] 实时股价数据加载成功

---

## 下一步

- [配置真实数据接入](./configuration.md) - 30 分钟接入真实征信数据
- [查看 FAQ 常见问题](./faq.md) - 故障排查

---

## 故障速查

| 现象 | 可能原因 | 解决方案 |
|:-----|:---------|:---------|
| `ModuleNotFoundError` | 依赖未安装 | `pip install -r requirements.txt` |
| XGBoost 安装失败 | 编译环境缺失 | `pip install xgboost --no-binary :all:` |
| Streamlit 端口占用 | 8501 被占用 | `streamlit run app.py --server.port 8502` |
| SHAP 图表空白 | matplotlib 后端问题 | 重启 Streamlit |

---

*最后更新: 2026-02-08*
