# 术语表 (Glossary)

## 通用术语

| 术语 | 英文 | 定义 |
|:-----|:-----|:-----|
| 人口稳定性指数 | Population Stability Index (PSI) | 衡量分布变化的指标，<0.1 稳定，>0.25 不稳定 |
| 特征稳定性指数 | Characteristic Stability Index (CSI) | 特征层面的 PSI |
| t 统计量 | t-statistic | 检验均值显著性的统计量 |
| p 值 | p-value | 显著性水平，<0.05 通常认为统计显著 |

## credit-one 专用术语

| 术语 | 英文 | 定义 |
|:-----|:-----|:-----|
| PD | Probability of Default | 违约概率 |
| LGD | Loss Given Default | 违约损失率 |
| EAD | Exposure At Default | 违约风险敞口 |
| ECL | Expected Credit Loss | 预期信用损失 |
| K-S 统计量 | Kolmogorov-Smirnov Statistic | 衡量模型区分能力的指标 |
| Gini | Gini Coefficient | 基尼系数，AUC 的线性变换 (2×AUC-1) |
| AR | Accuracy Ratio | 准确率比率 |
| ECE | Expected Calibration Error | 期望校准误差，衡量概率校准度 |
| SHAP | SHapley Additive exPlanations | 基于博弈论的特征归因方法 |
| OOT | Out-of-Time Validation | 时序外验证，用未来数据测试模型稳定性 |
| OOS | Out-of-Sample Validation | 样本外验证 |
| PDO | Points to Double the Odds | 评分卡中 odds 翻倍所需分数 |
| 评分卡 | Scorecard | 将模型输出转换为标准分数的线性模型 |
| 特征工程 | Feature Engineering | 从原始数据构建模型输入特征的过程 |
| 模型漂移 | Model Drift | 模型性能随时间下降的现象 |
| 数据漂移 | Data Drift | 输入数据分布变化 |
| 概念漂移 | Concept Drift | 特征与目标关系变化 |

## 监管框架术语

| 术语 | 说明 |
|:-----|:-----|
| Basel III | 巴塞尔协议 III，银行资本充足率监管框架 |
| IFRS 9 | 国际财务报告准则第 9 号，金融工具会计处理 |
| SR 11-7 | 美联储模型风险管理监管指引 |
| MRM | Model Risk Management，模型风险管理 |
| SICR | Significant Increase in Credit Risk，信用风险显著增加 |
| 三道防线 | Three Lines of Defense，风险管理组织架构 |

## 参考

- [项目限制说明](limitations.md)
- [模型治理框架](../model_governance.md)
- [评分卡校准](../scorecard_calibration.md)
