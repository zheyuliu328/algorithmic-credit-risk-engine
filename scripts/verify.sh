#!/usr/bin/env bash
# Run checks without deleting or replacing existing data or artifacts.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/verify.py" "$@"
