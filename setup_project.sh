#!/bin/bash
# Script to create the full, commented folder structure for the Crypto-Quant-MVP project.
# This structure assumes a microservices architecture where each Python service
# manages its own isolated virtual environment.

set -e

echo "🚀 Creating Crypto-Quant-MVP folder structure..."
echo "-------------------------------------------------"

# --- Core Project & Infrastructure Files ---
echo "⚙️  Setting up core project and infrastructure directories..."
mkdir -p .github/workflows
mkdir -p .vscode
mkdir -p infra/kubernetes/base
mkdir -p infra/kubernetes/charts
mkdir -p infra/kubernetes/apps
mkdir -p infra/terraform

# --- Development & Research Directories ---
echo "🔬 Setting up development and research directories..."
mkdir -p dags
mkdir -p notebooks
mkdir -p spark-jobs

# --- Service Directories (The Microservices) ---
echo "🛠️  Setting up individual service directories..."
# API Service (FastAPI)
mkdir -p services/api/app
mkdir -p services/api/tests
# Frontend Service (React/TS)
mkdir -p services/frontend/public
mkdir -p services/frontend/src/components
# Ingestor Service (Python)
mkdir -p services/ingestor/app
mkdir -p services/ingestor/tests
# Inference Service (PyTorch/Ray)
mkdir -p services/inference/app
# Training Service (PyTorch)
mkdir -p services/training/app

# --- Create Placeholder and Initial Files ---
echo "📄 Creating placeholder and initial configuration files..."
touch .github/workflows/ci.yml .github/workflows/cd.yml
touch .vscode/launch.json .vscode/settings.json
touch dags/__init__.py dags/datasets.py dags/sentiment_analysis_dag.py
touch infra/kubernetes/apps/api-deployment.yaml infra/kubernetes/apps/ingestor-deployment.yaml
touch infra/terraform/main.tf infra/terraform/variables.tf
touch notebooks/1_initial_data_exploration.ipynb notebooks/2_strategy_backtesting.ipynb
touch services/api/app/__init__.py services/api/app/main.py
touch services/api/tests/test_api_endpoints.py
touch services/api/Dockerfile services/api/pyproject.toml
touch services/frontend/public/.gitkeep
touch services/frontend/src/App.tsx services/frontend/src/components/.gitkeep
touch services/frontend/Dockerfile services/frontend/index.html services/frontend/package.json
touch services/ingestor/app/main.py services/ingestor/tests/.gitkeep services/ingestor/pyproject.toml
touch services/inference/app/.gitkeep services/inference/pyproject.toml
touch services/training/app/.gitkeep services/training/pyproject.toml
touch spark-jobs/ohlcv_aggregator.py
touch .dockerignore README.md docker-compose.dev.yml

# --- Create a comprehensive .gitignore file ---
echo "📝 Creating .gitignore file..."
echo "# Python Virtual Environments
.venv/
venv/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# IDE & OS specific
.vscode/
.idea/
.DS_Store
*.swp

# Build artifacts & local data
dist/
build/
*.egg-info/
/mlruns/
/artifacts/
/data/

# Node.js dependencies
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Logs
*.log
" > .gitignore

echo "-------------------------------------------------"
echo "✅ Project structure created successfully!"
echo ""
echo "🔥 IMPORTANT NEXT STEPS:"
echo "--------------------------"
echo "This project uses multiple, isolated virtual environments."
echo "Initialize each one as needed:"
echo ""
echo "1. For the API service:"
echo "   cd services/api && uv venv"
echo ""
echo "2. For the data ingestor:"
echo "   cd services/ingestor && uv venv"
echo ""
echo "3. For ML training:"
echo "   cd services/training && uv venv"
echo ""
echo "4. For data science notebooks:"
echo "   cd notebooks && uv venv"
echo ""
echo "...and so on for each Python application directory."
echo "Happy building!"
