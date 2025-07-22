#!/bin/bash

# Download required model and datasets for ML2025 HW1 RAG

set -e  # Exit on any error

echo "📥 Downloading ML2025 HW1 RAG Assets..."
echo "======================================"

# Download LLaMA 3.1 8B quantized model (~8GB)
echo "📦 Downloading LLaMA 3.1 8B Instruct model (Q8_0, ~8GB)..."
if [ ! -f "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf" ]; then
    wget -c https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
    echo "✅ Model downloaded successfully"
else
    echo "✅ Model already exists, skipping download"
fi

# Download question datasets
echo "📄 Downloading question datasets..."
if [ ! -f "public.txt" ]; then
    wget -c https://www.csie.ntu.edu.tw/~ulin/public.txt
    echo "✅ Public dataset downloaded"
else
    echo "✅ Public dataset already exists"
fi

if [ ! -f "private.txt" ]; then
    wget -c https://www.csie.ntu.edu.tw/~ulin/private.txt
    echo "✅ Private dataset downloaded"
else
    echo "✅ Private dataset already exists"
fi

echo "✅ All assets downloaded successfully!"