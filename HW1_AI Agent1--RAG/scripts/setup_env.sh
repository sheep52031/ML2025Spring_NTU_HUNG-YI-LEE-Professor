#!/bin/bash

# ML2025 HW1 RAG - Modern Environment Setup Script
# This script sets up the environment using the modern Conda + uv approach

set -e  # Exit on any error

echo "🚀 Setting up ML2025 HW1 RAG Environment (Modern Approach)"
echo "============================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if uv is installed globally
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (modern Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create conda environment from environment.yml
echo "🔧 Creating conda environment from environment.yml..."
conda env create -f environment.yml --force

# Activate the environment
echo "🔄 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml2025spring-rag

# Install Python dependencies with uv
echo "📦 Installing Python dependencies with uv..."
uv pip install -e .

# Download model and datasets
echo "📥 Downloading model and datasets..."
./scripts/download_assets.sh

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate ml2025spring-rag"
echo ""
echo "To run the notebook:"
echo "  jupyter notebook 'HW1_AI Agent1 -- RAG.ipynb'"