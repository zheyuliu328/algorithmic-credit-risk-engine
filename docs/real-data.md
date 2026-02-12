# Real Data Guide

## 真实数据接入路径

### 支持的数据格式

CSV 文件，必须包含以下字段：

| 字段名 | 类型 | 说明 |
|:-------|:-----|:-----|
| loan_amnt | float | 贷款金额 |
| term | string | 期限 (e.g., "36 months") |
| int_rate | float | 利率 |
| installment | float | 月供 |
| annual_inc | float | 年收入 |
| dti | float | 债务收入比 |
| earliest_cr_line | string | 最早信用记录日期 |
| open_acc | int | 开放账户数 |
| pub_rec | int | 公共记录数 |
| revol_bal | float | 循环信用余额 |
| revol_util | float | 循环信用利用率 |
| total_acc | int | 总账户数 |

### 快速开始

```bash
# 1. 验证数据格式
make run-real CSV=path/to/your/data.csv

# 2. 仅验证不处理
python scripts/run_real.py path/to/data.csv --validate-only
```

### 示例 CSV

```csv
loan_amnt,term,int_rate,installment,annual_inc,dti,earliest_cr_line,open_acc,pub_rec,revol_bal,revol_util,total_acc
10000.0,36 months,10.5,324.5,50000.0,15.2,Jan-2010,5,0,5000.0,25.0,15
15000.0,60 months,12.0,333.0,75000.0,20.1,Mar-2008,8,1,8000.0,35.0,20
```

### 常见错误与修复

| 错误 | 原因 | 修复 |
|:-----|:-----|:-----|
| Missing columns | CSV 缺少必需字段 | 检查字段名拼写 |
| File not found | 路径错误 | 使用绝对路径或检查相对路径 |
| dti must be numeric | 数据类型错误 | 确保数值字段不含文本 |

### 输出工件

运行后生成：
- `artifacts/scoring_report_{run_id}.json` - 评分报告
- `artifacts/scoring_output_{run_id}.csv` - 带评分结果的完整数据
