# Troubleshooting Guide - 常见故障与修复

> 10 条常见失败与一行修复方案

---

## 🔴 严重错误（阻止运行）

### 1. ModuleNotFoundError: No module named 'xgboost'
**现象**: 运行 `python model_validation.py` 时报错
```
ModuleNotFoundError: No module named 'xgboost'
```
**修复**:
```bash
pip install -r requirements.txt
```

### 2. XGBoost 安装失败 / 编译错误
**现象**: pip 安装 xgboost 时编译失败
**修复**:
```bash
pip install xgboost --no-binary :all:  # 从源码编译
# 或
conda install -c conda-forge xgboost   # 使用 conda
```

### 3. SHAP 图表空白或无法显示
**现象**: Streamlit 界面中 SHAP 图表空白
**修复**:
```bash
pip install matplotlib --upgrade
export MPLBACKEND=Agg  # Linux/Mac
set MPLBACKEND=Agg     # Windows
```

---

## 🟡 警告错误（功能受限）

### 4. Streamlit 端口被占用
**现象**: `streamlit run app.py` 报错端口 8501 被占用
**修复**:
```bash
streamlit run app.py --server.port 8502
```

### 5. 内存不足警告
**现象**: 运行模型验证时内存不足
**修复**:
```bash
# 减少数据集大小
python model_validation.py --sample-size 1000
```

### 6. 随机种子不一致导致结果不同
**现象**: 多次运行结果略有差异
**修复**:
```bash
# 检查 .env 中 RANDOM_SEED 设置
grep RANDOM_SEED .env
# 应输出: RANDOM_SEED=42
```

---

## 🟢 环境问题

### 7. Python 版本不兼容
**现象**: 运行时报语法错误或类型提示错误
**修复**:
```bash
# 检查 Python 版本
python --version  # 需要 3.9+
# 使用 pyenv 切换版本
pyenv install 3.9.0
pyenv local 3.9.0
```

### 8. 数据库文件损坏
**现象**: SQLite 报错 "database disk image is malformed"
**修复**:
```bash
# 删除损坏的数据库并重新运行
rm -f credit_risk.db psi_monitoring.db
python model_validation.py
```

### 9. 日志目录不存在
**现象**: 运行时报 "No such file or directory: './logs/'"
**修复**:
```bash
mkdir -p logs
```

### 10. 权限错误（Linux/Mac）
**现象**: Permission denied 错误
**修复**:
```bash
chmod +x run.sh
./run.sh
```

---

## 快速诊断命令

```bash
# 检查环境
python -c "import xgboost, shap, streamlit; print('OK')"

# 检查数据库
sqlite3 credit_risk.db ".tables"

# 检查日志
ls -la logs/
```

---

*最后更新: 2026-02-08*
