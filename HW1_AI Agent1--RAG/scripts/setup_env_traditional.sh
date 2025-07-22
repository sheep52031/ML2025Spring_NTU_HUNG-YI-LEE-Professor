#!/bin/bash

# ML2025 HW1 RAG - Traditional Environment Setup Script
# This script sets up the environment using traditional conda + pip approach

set -e  # Exit on any error

echo "🔧 Setting up ML2025 HW1 RAG Environment (Traditional Approach)"
echo "================================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment from environment.yml
echo "🔧 Creating conda environment from environment.yml..."
conda env create -f environment.yml --force

# Activate the environment
echo "🔄 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml2025spring-rag

# Install Python dependencies with pip
echo "📦 Installing Python dependencies with pip..."
pip install -r requirements.txt

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