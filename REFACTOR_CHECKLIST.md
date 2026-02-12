# Credit One - 文档产品化修改清单

## 修改目标
将 Credit One 项目文档重构为标准化用户路径文档，确保用户能在 3/10/30 分钟内完成上手、跑通和真实接入。

---

## 一、README.md 重构

**文件路径**: `credit-one/README.md`

**修改内容**:

```markdown
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
```

---

## 二、新建 docs/quickstart.md

**文件路径**: `credit-one/docs/quickstart.md`

**内容**:

```markdown
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
```

---

## 三、新建 docs/configuration.md

**文件路径**: `credit-one/docs/configuration.md`

**内容**:

```markdown
# Configuration Guide - 30 分钟真实接入

> 本指南帮助你接入真实征信数据，完成字段映射和模型配置。

---

## 前置要求

- 已完成 [Quickstart](./quickstart.md)
- 真实数据源访问权限（征信数据或内部数据）
- 了解数据表结构

---

## 一、数据源配置

### 1.1 支持的数据源

项目设计支持以下数据源:

| 数据源 | 类型 | 接入方式 |
|:-------|:-----|:---------|
| 百行征信 | 征信报告 | API / 数据文件 |
| 央行征信 | 征信报告 | API / 数据文件 |
| 运营商数据 | 行为数据 | API |
| 电商数据 | 交易数据 | API / 数据文件 |
| 内部数据 | 业务数据 | 数据库 / CSV |

### 1.2 配置文件

复制并编辑配置:

```bash
cp config/config.example.yaml config/config.yaml
```

### 1.3 配置数据连接

编辑 `config/config.yaml`:

```yaml
data_sources:
  pboc:
    type: "api"
    endpoint: "https://api.pboc.gov.cn/credit"
    api_key: "${PBOC_API_KEY}"
  
  internal:
    type: "database"
    driver: "postgresql"
    host: "localhost"
    port: 5432
    database: "credit_db"
    username: "${DB_USER}"
    password: "${DB_PASS}"  # Use environment variable
```

---

## 二、字段映射规范

### 2.1 征信数据字段映射

| 数据源 | 源字段 | 内部字段 | 说明 |
|:-------|:-------|:---------|:-----|
| 百行征信 | credit_score | pboc_score | 征信评分 |
| 百行征信 | overdue_count_12m | overdue_12m | 12个月逾期次数 |
| 百行征信 | total_credit_limit | credit_limit | 总授信额度 |
| 百行征信 | utilization_rate | utilization | 额度使用率 |
| 央行征信 | query_count_3m | query_3m | 3个月查询次数 |
| 运营商 | avg_call_duration | call_duration | 平均通话时长 |
| 运营商 | night_activity_ratio | night_ratio | 夜间活跃度 |

### 2.2 目标变量映射

| 源字段 | 内部字段 | 说明 |
|:-------|:---------|:-----|
| default_flag | target | 是否违约 (0/1) |
| default_date | default_date | 违约日期 |
| loan_amount | loan_amount | 贷款金额 |
| loan_term | loan_term | 贷款期限 |

### 2.3 自定义字段映射

编辑 `transform_logic.sql`:

```sql
-- 百行征信字段映射
SELECT 
    customer_id,
    credit_score as pboc_score,
    overdue_count_12m as overdue_12m,
    total_credit_limit as credit_limit,
    (used_credit / total_credit_limit) as utilization
FROM pboc_credit_report

UNION ALL

-- 内部数据字段映射
SELECT 
    customer_id,
    NULL as pboc_score,
    historical_overdue as overdue_12m,
    approved_limit as credit_limit,
    current_balance / approved_limit as utilization
FROM internal_credit_data
```

---

## 三、数据接入步骤

### 3.1 准备数据文件

```bash
# 方式1: 使用 CSV 文件
cp your_data.csv data/sme_credit_data.csv

# 方式2: 配置数据库连接
# 编辑 config/config.yaml
```

### 3.2 修改数据加载逻辑

编辑 `sme_credit_explainability.py`:

```python
def load_real_data():
    """加载真实数据"""
    # 方式1: 从 CSV 加载
    df = pd.read_csv('data/sme_credit_data.csv')
    
    # 方式2: 从数据库加载
    import psycopg2
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM credit_data", conn)
    
    return df

# 替换合成数据生成
def generate_synthetic_sme_data():
    return load_real_data()
```

### 3.3 配置特征工程

编辑特征配置:

```python
# 在 sme_credit_explainability.py 中
FEATURE_COLUMNS = [
    'pboc_score',
    'overdue_12m',
    'credit_limit',
    'utilization',
    'query_3m',
    # 添加自定义特征
    'your_custom_feature'
]
```

---

## 四、模型配置

### 4.1 XGBoost 参数

编辑模型配置:

```python
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

### 4.2 Scorecard 校准

编辑评分卡配置:

```python
class ScorecardConfig:
    BASE_SCORE = 600
    PDO = 20  # Points to Double the Odds
    ODDS_AT_BASE = 50  # 1:50
```

---

## 五、常见失败点

### 5.1 数据加载失败

**现象**: `FileNotFoundError` 或数据库连接错误

**排查步骤**:
1. 检查文件路径是否正确
2. 确认数据库连接参数
3. 验证网络连通性
4. 检查认证信息

### 5.2 字段映射错误

**现象**: `KeyError` 或模型训练失败

**排查步骤**:
1. 检查 CSV 列名是否与映射一致
2. 确认大小写敏感
3. 检查是否有空格或特殊字符

### 5.3 数据类型错误

**现象**: `TypeError` 或 `ValueError`

**解决方案**:
```python
# 添加类型转换
df['pboc_score'] = pd.to_numeric(df['pboc_score'], errors='coerce')
df['overdue_12m'] = df['overdue_12m'].fillna(0).astype(int)
```

### 5.4 模型性能下降

**现象**: AUC 显著低于预期

**排查步骤**:
1. 检查数据质量（缺失值比例）
2. 验证目标变量分布
3. 检查特征相关性
4. 确认训练/测试集划分合理

### 5.5 SHAP 解释失败

**现象**: SHAP 图表空白或报错

**解决方案**:
```bash
# 重新安装 shap
pip uninstall shap
pip install shap>=0.41.0

# 重启 Streamlit
streamlit run app.py
```

---

## 六、验证清单

接入完成后，验证以下项目:

- [ ] 数据加载成功，记录数符合预期
- [ ] 字段映射正确，无 KeyError
- [ ] 模型训练完成，AUC > 0.80
- [ ] OOT 验证通过，AUC 下降 < 0.05
- [ ] PSI < 0.25（稳定）
- [ ] Streamlit 界面正常显示
- [ ] SHAP 解释正常生成

---

## 七、生产环境建议

### 7.1 数据安全

- 敏感数据加密存储
- 使用密钥管理服务
- 限制数据访问权限
- 定期审计数据使用

### 7.2 模型监控

- 每日监控 PSI 指标
- 每周检查模型性能
- 每月审查特征分布
- 设置自动告警

### 7.3 部署架构

```
[数据源] → [ETL] → [特征存储] → [模型服务] → [API/界面]
                ↓
           [监控/告警]
```

---

*最后更新: 2026-02-08*
```

---

## 四、新建 docs/faq.md

**文件路径**: `credit-one/docs/faq.md`

**内容**:

```markdown
# FAQ - 常见问题

---

## 安装问题

### Q: 运行 `model_validation.py` 报错 "No module named 'xgboost'"

**A**: 安装 XGBoost 依赖:
```bash
pip install xgboost>=1.7.0
# 或
pip install -r requirements.txt
```

### Q: XGBoost 安装失败（编译错误）

**A**: 使用预编译版本:
```bash
# macOS
pip install xgboost --no-binary :all:

# 或使用 conda
conda install -c conda-forge xgboost
```

### Q: Python 版本要求

**A**: 需要 Python 3.9+。检查版本:
```bash
python --version
```

---

## 运行问题

### Q: Streamlit 界面无法加载实时股价

**A**: 检查网络连接，或修改 `app.py` 使用本地数据:
```python
# 注释掉实时数据获取
# stock = yf.Ticker(ticker)
```

### Q: SHAP 解释图显示为空白

**A**: 确保已安装 shap 并重启 Streamlit:
```bash
pip install shap>=0.41.0
streamlit run app.py
```

### Q: 模型验证报告中的 AUC 阈值是多少?

**A**: 
- AUC Degradation < 0.05（可接受）
- PSI Score < 0.25（稳定）
- ECE < 0.05（校准良好）

### Q: 如何添加新的风险特征?

**A**: 修改 `sme_credit_explainability.py`:
```python
BUSINESS_INSIGHTS["new_feature"] = {
    "name": "New Feature",
    "threshold": 0.5,
    "why_risk": "Explanation for high risk",
    "why_safe": "Explanation for low risk",
}

# 添加到特征列表
FEATURE_COLUMNS.append('new_feature')
```

---

## 数据问题

### Q: 如何接入真实征信数据?

**A**: 见 [configuration.md](./configuration.md) 的数据接入章节。

### Q: 合成数据与真实数据差异大怎么办?

**A**: 调整合成数据生成参数:
```python
def generate_synthetic_sme_data():
    # 调整分布参数以匹配真实数据
    n_samples = 10000  # 增加样本数
    # 修改特征分布...
```

### Q: 数据文件格式要求?

**A**: 
- 格式: CSV
- 编码: UTF-8
- 分隔符: 逗号
- 首行: 列名

---

## 模型问题

### Q: 模型 AUC 太低怎么办?

**A**: 
1. 检查数据质量
2. 增加特征工程
3. 调整模型参数
4. 增加训练数据量

### Q: OOT AUC 下降太多怎么办?

**A**: 
- 检查时间划分是否合理
- 验证数据分布是否变化
- 考虑添加时间特征
- 可能需要重新训练

### Q: PSI 过高怎么办?

**A**: 
- PSI < 0.1: 稳定，无需处理
- 0.1 < PSI < 0.25: 警告，监控趋势
- PSI > 0.25: 临界，建议重新训练

---

## 界面问题

### Q: Streamlit 端口被占用

**A**: 更换端口:
```bash
streamlit run app.py --server.port 8502
```

### Q: 如何部署到服务器?

**A**: 
```bash
# 使用 nohup
nohup streamlit run app.py --server.port 8501 &

# 或使用 systemd 配置服务
```

### Q: 报告中文乱码

**A**: 安装中文字体:
```bash
# macOS
brew install font-wqy-zenhei

# Ubuntu
sudo apt-get install fonts-wqy-zenhei
```

---

## 其他问题

### Q: 如何导出模型?

**A**: 
```python
import joblib
# 保存模型
joblib.dump(model, 'credit_model.pkl')
# 加载模型
model = joblib.load('credit_model.pkl')
```

### Q: 如何批量预测?

**A**: 
```python
df = pd.read_csv('batch_data.csv')
predictions = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
df['pd_score'] = predictions
df.to_csv('predictions.csv', index=False)
```

### Q: 项目是否支持其他模型?

**A**: 支持。修改 `sme_credit_explainability.py`:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
```

---

*最后更新: 2026-02-08*
```

---

## 五、新建 docs 目录

**命令**:
```bash
mkdir -p credit-one/docs
```

---

## 六、文件创建/修改清单总结

| 文件路径 | 操作 | 说明 |
|:---------|:-----|:-----|
| `credit-one/README.md` | 修改 | 重构为标准化结构 |
| `credit-one/docs/quickstart.md` | 新建 | 10 分钟跑通指南 |
| `credit-one/docs/configuration.md` | 新建 | 30 分钟接入配置 |
| `credit-one/docs/faq.md` | 新建 | 常见问题解答 |
| `credit-one/docs/` | 新建目录 | 文档目录 |

---

## 关键纠偏落实

1. **监管合规描述**: 
   - 删除了 "Basel III / IFRS 9 compliant"、"SR 11-7 Compliant" 等无事实支持的表述
   - 删除了 "符合监管要求" 等绝对化描述
   - 统一使用 "面向风险建模与研究的工具" 作为定位

2. **移除夸大描述**:
   - 删除了 "Production-grade"、"Production Ready" 等词汇
   - 删除了 "实时 Inference"、"2500 TPS" 等未经验证的性能声明
   - 删除了 "合规" 相关的徽章

3. **事实源支持**:
   - 所有指标均来自实际运行输出（AUC、K-S、PSI 等）
   - 明确标注当前使用合成数据，真实数据需自行接入
