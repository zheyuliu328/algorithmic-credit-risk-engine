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
