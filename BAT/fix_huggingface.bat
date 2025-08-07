@echo off
echo ========================================
echo  Fix HuggingFace Hub Compatibility Issue
echo ========================================
echo.

cd /d D:\genAI\LangChain-RAG

echo [1/4] Installing new langchain-huggingface package...
pip install langchain-huggingface --upgrade
echo.

echo [2/4] Updating sentence-transformers...
pip install sentence-transformers --upgrade
echo.

echo [3/4] Fixing huggingface-hub compatibility...
pip install huggingface-hub==0.24.7 --upgrade
echo.

echo [4/4] Testing the fix...
python -c "from langchain_huggingface import HuggingFaceEmbeddings; print('✅ langchain-huggingface OK')" 2>nul
if errorlevel 1 (
    echo langchain-huggingface not available, trying sentence-transformers...
    python -c "from sentence_transformers import SentenceTransformer; print('✅ sentence-transformers OK')"
)

echo.
echo Testing fixed embeddings...
python fixed_embeddings.py
echo.

echo ========================================
echo  Fix completed! You can now run:
echo  streamlit run streamlit_app.py
echo ========================================
pause
