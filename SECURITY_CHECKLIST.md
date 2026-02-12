# Credit-One 安全改造清单

## 文件修改清单

### 1. src/utils/guardrails.py (新增)
- **路径**: `credit-one/src/utils/guardrails.py`
- **操作**: 复制根目录 guardrails.py

### 2. src/utils/secrets.py (新增)
- **路径**: `credit-one/src/utils/secrets.py`
- **操作**: 复制根目录 secrets.py

### 3. src/utils/data_boundary.py (新增)
- **路径**: `credit-one/src/utils/data_boundary.py`
- **操作**: 复制根目录 data_boundary.py

### 4. config/config.yaml (修改)
- **路径**: `credit-one/config/config.yaml`
- **修改内容**:

```yaml
version: "2.0.0"

# 数据配置
data:
  input_path: "./data"
  output_path: "./artifacts"
  format: "csv"
  
# 模型配置
model:
  name: "XGBoost_PD_Model"
  version: "v2.0"
  random_seed: 42
  
# 评分卡配置
scorecard:
  base_score: 600
  base_odds: 50
  pdo: 20
  
# PSI监控配置
psi:
  stable_threshold: 0.10
  warning_threshold: 0.25
  
# 日志配置
logging:
  level: "INFO"
  format: "json"
  output: "./logs"
  
# 运行配置
run:
  dry_run: false
  confirm: false
  verbose: false
  debug: false
  
# 安全配置
security:
  # 危险操作确认
  require_confirm: true
  # 审计日志
  audit_enabled: true
  # 数据校验
  validate_input: true
  # 路径白名单
  allowed_paths:
    - "./data"
    - "./artifacts"
    - "./logs"
    - "./backups"
```

### 5. config/validator.py (修改)
- **路径**: `credit-one/config/validator.py`
- **完整内容**:

```python
"""
配置校验器 - 加载和校验配置文件
禁止从配置文件读取 Secrets
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigValidator:
    """配置校验器"""
    
    # 禁止在配置中出现的敏感字段
    FORBIDDEN_FIELDS = [
        'password', 'secret', 'api_key', 'apikey', 'token',
        'private_key', 'access_key', 'secret_key'
    ]
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
    
    def load(self) -> Dict[str, Any]:
        """加载并校验配置"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_security()
        self._validate_paths()
        
        return self.config
    
    def _validate_security(self):
        """校验安全配置"""
        # 检查是否包含敏感字段
        config_str = str(self.config).lower()
        for field in self.FORBIDDEN_FIELDS:
            if field in config_str:
                # 检查是否有实际值（不是示例值）
                for section in self.config.values():
                    if isinstance(section, dict):
                        for key, value in section.items():
                            if field in key.lower() and value and value not in ['YOUR_', 'example', 'test']:
                                raise ValueError(
                                    f"Security violation: '{key}' found in config file. "
                                    f"Secrets must be stored in environment variables only."
                                )
    
    def _validate_paths(self):
        """校验路径配置"""
        security = self.config.get('security', {})
        allowed_paths = security.get('allowed_paths', [])
        
        # 确保所有路径都在白名单内
        for key in ['input_path', 'output_path']:
            path = self.config.get('data', {}).get(key, '')
            if path and not any(str(path).startswith(p) for p in allowed_paths):
                raise ValueError(f"Path '{path}' is not in allowed paths: {allowed_paths}")


def load_config() -> Dict[str, Any]:
    """便捷函数：加载配置"""
    validator = ConfigValidator()
    return validator.load()
```

### 6. pipeline.py (修改)
- **路径**: `credit-one/pipeline.py`
- **修改内容**:

```python
# 在文件顶部添加:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.guardrails import (
    DangerousOpGuard, PathValidator, AuditLogger, 
    require_confirm, validate_path
)
from src.utils.data_boundary import validate_loan_data, FieldSchema, FieldType
from config.validator import load_config

# 修改 create_database 函数:
@require_confirm("database.delete")
def create_database(confirm: bool = False):
    """Initialize SQLite database"""
    guard = DangerousOpGuard()
    
    if os.path.exists(DB_NAME):
        if not guard.check("database.delete", DB_NAME, confirm_flag=confirm):
            print("Database creation cancelled")
            return None
        os.remove(DB_NAME)
        print(f"Removed existing {DB_NAME}")
    
    conn = sqlite3.connect(DB_NAME)
    print(f"Created database: {DB_NAME}")
    return conn

# 修改 step1_extract_and_load:
def step1_extract_and_load(conn, n_samples=50000, validate: bool = True):
    """添加数据校验"""
    # ... 原有代码 ...
    
    if validate:
        # 校验数据
        validation_result = validate_loan_data(df_raw)
        if not validation_result.is_valid:
            print("[ERROR] Data validation failed:")
            for error in validation_result.errors[:10]:  # 只显示前10个
                print(f"  - {error}")
            raise ValueError("Data validation failed")
    
    # ... 原有代码 ...
```

### 7. main.py (修改)
- **路径**: `credit-one/main.py`
- **修改内容**:

```python
# 添加参数支持:
def main():
    """Main pipeline execution"""
    # 加载配置
    config = load_config()
    
    # 解析命令行参数
    use_real_data = '--real-data' in sys.argv or '--lending-club' in sys.argv
    n_samples = 10000
    confirm = '--confirm' in sys.argv
    dry_run = '--dry-run' in sys.argv
    
    if '--samples' in sys.argv:
        idx = sys.argv.index('--samples')
        if idx + 1 < len(sys.argv):
            n_samples = int(sys.argv[idx + 1])
    
    # 设置环境变量
    if dry_run:
        os.environ['GUARDRAILS_DRY_RUN'] = 'true'
    if confirm:
        os.environ['GUARDRAILS_CONFIRM'] = 'true'
    
    # ... 原有代码 ...
    
    # 使用确认标志创建数据库
    conn = create_database(confirm=confirm)
    if conn is None:
        print("Operation cancelled by user")
        return
```

### 8. .env.example (新增)
- **路径**: `credit-one/.env.example`

```bash
# Kaggle API Credentials
# 获取地址: https://www.kaggle.com/settings/account
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key

# 注意: 复制此文件为 .env 并填入真实值
# .env 文件已添加到 .gitignore，不会被提交
```

### 9. .gitignore (修改)
- **路径**: `credit-one/.gitignore`
- **添加内容**:

```gitignore
# Secrets
.env
.env.local
.env.*.local
*.pem
*.key
kaggle.json

# Data (large files)
data/raw/*.csv
data/raw/*.parquet
!data/raw/.gitkeep

# Database
*.db
*.db-journal
*.db-wal

# Logs
logs/
*.log

# Backups
backups/

# Artifacts
artifacts/models/*.pkl
artifacts/models/*.joblib
```

### 10. pyproject.toml (修改)
- **路径**: `credit-one/pyproject.toml`
- **添加内容**:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "ruff>=0.0.200",
    "mypy>=0.990",
    "pre-commit>=2.20.0",
    "bandit>=1.7.0",
]

[tool.bandit]
exclude_dirs = ["tests", "artifacts"]
skips = ["B101"]
```

### 11. .pre-commit-config.yaml (新增)
- **路径**: `credit-one/.pre-commit-config.yaml`
- **内容**: 同 FCT 配置

### 12. .gitleaks.toml (新增)
- **路径**: `credit-one/.gitleaks.toml`
- **操作**: 复制根目录 .gitleaks.toml

### 13. scripts/backup.sh (新增)
- **路径**: `credit-one/scripts/backup.sh`
- **内容**: 同 FCT backup.sh

### 14. docs/SECURITY.md (新增)
- **路径**: `credit-one/docs/SECURITY.md`

```markdown
# Credit-One 安全指南

## Secrets 管理
- Kaggle 凭证通过环境变量读取
- 配置文件禁止存储任何敏感信息

## 危险操作
- 数据库删除需要 `--confirm` 参数
- 使用 `--dry-run` 预览操作

## 数据校验
- 所有贷款数据通过 `validate_loan_data()` 校验
- 校验包括：字段类型、范围、缺失率

## 审计日志
- 所有操作记录在 `logs/audit_YYYYMMDD.log`
- 包含：操作类型、目标、时间、用户

## 回滚
```bash
# 数据库回滚
./scripts/rollback_db.sh credit_risk.db 20240208_120000

# 配置回滚
./scripts/rollback_config.sh config.yaml v1.0
```
```

## 实施步骤

1. **创建工具模块**
   ```bash
   mkdir -p credit-one/src/utils
   cp src/utils/guardrails.py credit-one/src/utils/
   cp src/utils/secrets.py credit-one/src/utils/
   cp src/utils/data_boundary.py credit-one/src/utils/
   ```

2. **修改配置文件**
   ```bash
   # 编辑 config/config.yaml 添加安全配置
   # 编辑 config/validator.py 添加配置校验
   ```

3. **修改主程序**
   ```bash
   # 编辑 pipeline.py 添加危险操作保护
   # 编辑 main.py 添加参数支持
   ```

4. **配置环境**
   ```bash
   cd credit-one
   cp .env.example .env
   # 编辑 .env 填入 Kaggle 凭证
   ```

5. **配置安全扫描**
   ```bash
   cp ../.gitleaks.toml .
   pre-commit install
   ```

6. **验证**
   ```bash
   # 测试危险操作确认
   python main.py --dry-run
   python main.py --confirm
   
   # 测试数据校验
   python pipeline.py
   
   # 测试 gitleaks
   gitleaks detect --source . --verbose
   ```
