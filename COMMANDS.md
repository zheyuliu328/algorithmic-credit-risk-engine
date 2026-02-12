# 🚀 Quick Command Reference

Essential commands for CreditOne V6.0 development and deployment.

---

## 📦 Development Commands

### Start Application
```bash
# Standard start
streamlit run app.py

# Specify port
streamlit run app.py --server.port 8501

# Headless mode (no browser auto-open)
streamlit run app.py --server.headless=true

# Background mode
nohup streamlit run app.py > streamlit.log 2>&1 &
```

### Stop Application
```bash
# Kill streamlit process
pkill -f "streamlit run app.py"

# Or find and kill by PID
ps aux | grep streamlit
kill <PID>
```

---

## 🧪 Testing Commands

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_credit_model.py -v

# Run specific test class
pytest tests/test_credit_model.py::TestPSICalculation -v

# Run specific test method
pytest tests/test_credit_model.py::TestPSICalculation::test_psi_identical_distributions -v
```

### Coverage Reports
```bash
# Terminal coverage report
pytest tests/ --cov=. --cov-report=term-missing

# HTML coverage report
pytest tests/ --cov=. --cov-report=html

# Open HTML report (macOS)
open htmlcov/index.html

# Open HTML report (Linux)
xdg-open htmlcov/index.html
```

### Watch Mode (Auto-run tests)
```bash
# Install pytest-watch
pip install pytest-watch

# Run in watch mode
ptw tests/ -- -v
```

---

## 🔧 Git Commands

### Basic Workflow
```bash
# Check status
git status

# Add all files
git add .

# Add specific file
git add app.py

# Commit with message
git commit -m "feat: add new feature"

# Push to remote
git push origin main
```

### Branching
```bash
# Create new branch
git checkout -b feature/new-feature

# Switch branch
git checkout main

# List branches
git branch -a

# Delete branch
git branch -d feature/old-feature
```

### Tagging & Releases
```bash
# Create annotated tag
git tag -a v6.0.0 -m "Release V6.0"

# Push tag to remote
git push origin v6.0.0

# List all tags
git tag -l

# Delete tag
git tag -d v6.0.0
git push origin :refs/tags/v6.0.0
```

### History & Logs
```bash
# View commit history
git log --oneline --graph --all

# View changes in last commit
git show

# View file history
git log --follow app.py

# Undo last commit (keep changes)
git reset --soft HEAD~1
```

---

## 🐍 Python Environment

### Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate
```

### Package Management
```bash
# Install dependencies
pip install -r requirements.txt

# Install single package
pip install streamlit

# Upgrade package
pip install --upgrade streamlit

# Uninstall package
pip uninstall streamlit

# List installed packages
pip list

# Export dependencies
pip freeze > requirements.txt

# Show package info
pip show streamlit
```

---

## 📊 Data & Model Commands

### Generate Test Data
```bash
# Create test CSV for PSI monitoring
python3 -c "
import pandas as pd
import numpy as np
np.random.seed(42)
df = pd.DataFrame({
    'revenue_growth': np.random.normal(0.12, 0.05, 100),
    'debt_to_asset_ratio': np.random.uniform(0.3, 0.9, 100),
    'cash_flow_volatility': np.random.uniform(0.5, 2.5, 100)
})
df.to_csv('test_production_data.csv', index=False)
print('✅ Created test_production_data.csv')
"
```

### Run Model Training
```bash
# Train models and generate reports
python3 sme_credit_explainability.py
```

---

## 🌐 Browser & Screenshots

### Open Application
```bash
# macOS
open http://localhost:8501

# Linux
xdg-open http://localhost:8501

# Windows
start http://localhost:8501
```

### Take Screenshots (macOS)
```bash
# Full screen
Cmd + Shift + 3

# Selected area
Cmd + Shift + 4

# Window
Cmd + Shift + 4, then Space
```

---

## 🔍 Debugging Commands

### Check Python Version
```bash
python3 --version
```

### Check Package Versions
```bash
pip show streamlit xgboost scikit-learn
```

### Find Port Usage
```bash
# Check if port 8501 is in use
lsof -i :8501

# Kill process on port 8501
lsof -ti :8501 | xargs kill -9
```

### View Logs
```bash
# View streamlit log (if running in background)
tail -f streamlit.log

# View last 50 lines
tail -n 50 streamlit.log
```

---

## 📁 File Operations

### Create Directories
```bash
# Create single directory
mkdir docs

# Create nested directories
mkdir -p docs/screenshots

# Create multiple directories
mkdir tests data models
```

### File Management
```bash
# Copy file
cp app.py app_backup.py

# Move/rename file
mv old_name.py new_name.py

# Delete file
rm file.py

# Delete directory
rm -rf directory/

# Find files
find . -name "*.py"

# Count lines of code
find . -name "*.py" | xargs wc -l
```

---

## 🚀 Deployment Commands

### Docker (Future)
```bash
# Build image
docker build -t creditone:v6.0 .

# Run container
docker run -p 8501:8501 creditone:v6.0

# Stop container
docker stop <container_id>
```

### Requirements Check
```bash
# Check for outdated packages
pip list --outdated

# Security audit
pip install safety
safety check
```

---

## 💡 Useful Aliases

Add these to your `~/.zshrc` or `~/.bashrc`:

```bash
# Quick start
alias creditone="cd /Users/zheyuliu/Documents/GitHub/algorithmic-credit-risk-engine && streamlit run app.py"

# Quick test
alias test-creditone="cd /Users/zheyuliu/Documents/GitHub/algorithmic-credit-risk-engine && pytest tests/ -v"

# Quick commit
alias gc="git add . && git commit -m"

# Quick push
alias gp="git push origin main"
```

---

## 📝 Notes

- Always activate virtual environment before running commands
- Use `python3` instead of `python` on macOS/Linux
- Check port availability before starting Streamlit
- Run tests before committing code
- Keep dependencies updated regularly

---

**Last Updated**: 2026-01-14
