@echo off
echo ========================================
echo   Final Fix - Compatible Versions
echo ========================================
echo.

cd /d D:\genAI\LangChain-RAG

echo [1/6] Uninstalling problematic packages...
pip uninstall sentence-transformers langchain langchain-community -y
echo.

echo [2/6] Installing LangChain suite...
pip install langchain==0.2.11 langchain-community==0.2.10
echo.

echo [3/6] Installing newer sentence-transformers...
echo Installing sentence-transformers 3.0+ (works with new huggingface-hub)...
pip install sentence-transformers>=3.0.0
echo.

echo [4/6] Checking huggingface-hub version...
pip show huggingface-hub
echo.

echo [5/6] Installing other essentials...
pip install streamlit==1.37.0 faiss-cpu==1.8.0 pypdf==4.3.1 python-dotenv==1.0.1 google-generativeai==0.7.2 langchain-google-genai==1.0.8
echo.

echo [6/6] Testing imports...
python -c "import langchain; print(f'✅ LangChain: {langchain.__version__}')" 2>nul || echo ❌ LangChain failed
python -c "import langchain_core; print(f'✅ LangChain Core: {langchain_core.__version__}')" 2>nul || echo ❌ LangChain Core failed
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('paraphrase-MiniLM-L3-v2'); print('✅ Sentence Transformers: Working!')" 2>nul || echo ❌ Sentence Transformers failed
python -c "import streamlit; print(f'✅ Streamlit: {streamlit.__version__}')" 2>nul || echo ❌ Streamlit failed
python -c "import faiss; print('✅ FAISS: OK')" 2>nul || echo ❌ FAISS failed
echo.

echo Testing embeddings with simple model...
python test_simple_embeddings.py
echo.

echo ========================================
echo   Fix completed!
echo   Run: streamlit run streamlit_app.py
echo ========================================
pause
