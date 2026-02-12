#!/bin/bash
set -e

echo "🚀 Credit Risk Engine - Quick Start"
echo "===================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies if needed
if ! python -c "import xgboost" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -q -r requirements.txt
fi
echo "✓ Dependencies installed"

# Run model validation
echo ""
echo "🔬 Running model validation..."
python model_validation.py

# Run PSI monitoring demo
echo ""
echo "📊 Running PSI monitoring demo..."
python psi_monitoring.py

# Summary
echo ""
echo "===================================="
echo "✅ Quick start complete!"
echo ""
echo "Next steps:"
echo "  • Run dashboard: streamlit run app.py"
echo "  • Run full pipeline: python main.py"
echo "  • View docs: cat README.md"
echo "===================================="
