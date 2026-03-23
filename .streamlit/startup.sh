#!/bin/bash
# Streamlit Cloud startup script
# 预下载模型，避免每次启动都下载

echo "🚀 Starting Streamlit app initialization..."

# 设置 Hugging Face 镜像（加速下载）
export HF_ENDPOINT=https://huggingface.co

# 预下载嵌入模型
echo "📥 Pre-downloading embedding model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')" 2>&1 || echo "Model download skipped or failed"

echo "✅ Initialization complete. Starting Streamlit..."

# 启动 Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
