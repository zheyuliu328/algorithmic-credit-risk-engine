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
    password: "${DB_PASSWORD}"
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
