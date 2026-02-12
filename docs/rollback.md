# Rollback Guide

## Version Rollback

### Pip Package Rollback

```bash
# Install specific version
pip install credit-risk-engine==1.9.0

# Check installed version
pip show credit-risk-engine
```

### Git Tag Rollback

```bash
# List available tags
git tag --list

# Checkout specific version
git checkout v1.9.0

# Verify make verify passes
make verify
```

### Docker Tag Rollback

```bash
# Pull specific version
docker pull credit-risk-engine:1.9.0

# Run with specific version
docker run credit-risk-engine:1.9.0
```

## Rollback演练记录

### 演练1: Git Tag 回滚

```bash
# 当前版本
git log -1 --oneline
# 8488646 Add run-real path with CSV validation and scoring

# 回滚到上一版本
git checkout ae81296

# 验证
make verify
# [OK] All checks passed!

# 回到最新版本
git checkout master
```

## 数据库回滚

```bash
# 备份当前数据库
cp data/credit_risk.db backups/credit_risk_$(date +%Y%m%d_%H%M%S).db

# 恢复旧版本
cp backups/credit_risk_20260208_010000.db data/credit_risk.db
```
