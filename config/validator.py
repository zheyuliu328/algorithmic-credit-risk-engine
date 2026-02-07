"""
Config Validator - Configuration schema validation
"""

import jsonschema
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

CONFIG_SCHEMA = {
    "type": "object",
    "required": ["version", "data", "model", "logging"],
    "properties": {
        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "data": {
            "type": "object",
            "required": ["input_path", "output_path"],
            "properties": {
                "input_path": {"type": "string"},
                "output_path": {"type": "string"},
                "format": {"enum": ["csv", "parquet", "json"]}
            }
        },
        "model": {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "random_seed": {"type": "integer"}
            }
        },
        "scorecard": {
            "type": "object",
            "properties": {
                "base_score": {"type": "integer", "minimum": 300, "maximum": 850},
                "base_odds": {"type": "integer", "minimum": 1},
                "pdo": {"type": "integer", "minimum": 1}
            }
        },
        "psi": {
            "type": "object",
            "properties": {
                "stable_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "warning_threshold": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "logging": {
            "type": "object",
            "required": ["level"],
            "properties": {
                "level": {"enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                "format": {"enum": ["json", "text"]},
                "output": {"type": "string"}
            }
        },
        "run": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean"},
                "confirm": {"type": "boolean"},
                "verbose": {"type": "boolean"},
                "debug": {"type": "boolean"}
            }
        }
    }
}


def validate_config(config_path: str) -> tuple[bool, Optional[str]]:
    """
    验证配置文件
    
    Returns:
        (is_valid, error_message)
    """
    path = Path(config_path)
    
    if not path.exists():
        return False, f"配置文件不存在: {config_path}"
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, f"YAML解析错误: {e}"
    
    try:
        jsonschema.validate(config, CONFIG_SCHEMA)
        return True, None
    except jsonschema.ValidationError as e:
        return False, f"配置错误 [{e.json_path}]: {e.message}"


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载并验证配置"""
    is_valid, error = validate_config(config_path)
    
    if not is_valid:
        raise ValueError(f"配置验证失败: {error}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    
    is_valid, error = validate_config(config_file)
    
    if is_valid:
        print(f"✅ 配置验证通过: {config_file}")
        sys.exit(0)
    else:
        print(f"❌ 配置验证失败: {error}")
        sys.exit(1)
