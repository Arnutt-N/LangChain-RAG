@echo off
echo ========================================
echo   Complete Dependency Fix for LangChain-RAG
echo ========================================
echo.

cd /d D:\genAI\LangChain-RAG

echo [STEP 1] Backing up current environment...
pip freeze > requirements_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%.txt
echo Backup saved.
echo.

echo [STEP 2] Uninstalling conflicting packages...
pip uninstall langchain langchain-core langchain-community langsmith langchain-huggingface -y
echo.

echo [STEP 3] Installing compatible versions...
echo.

echo Installing core LangChain packages (v0.2.x)...
pip install langchain==0.2.11 langchain-core==0.2.43 langchain-community==0.2.10 langsmith==0.1.99
echo.

echo Installing Google Generative AI...
pip install langchain-google-genai==1.0.8
echo.

echo Installing HuggingFace Hub (compatible version)...
pip install huggingface-hub==0.24.5
echo.

echo Installing Sentence Transformers (standalone)...
pip install sentence-transformers==2.2.2
echo.

echo Installing other dependencies...
pip install streamlit==1.37.0 faiss-cpu==1.8.0 pypdf==4.3.1 python-dotenv==1.0.1
echo.

echo [STEP 4] Verifying installation...
python -c "import langchain; print(f'✅ LangChain: {langchain.__version__}')"
python -c "import langchain_core; print(f'✅ LangChain Core: {langchain_core.__version__}')"
python -c "from sentence_transformers import SentenceTransformer; print('✅ Sentence Transformers: OK')"
python -c "import streamlit; print(f'✅ Streamlit: {streamlit.__version__}')"
echo.

echo [STEP 5] Testing embeddings...
python test_standalone_embeddings.py
echo.

echo ========================================
echo   Fix completed!
echo   Run: streamlit run streamlit_app.py
echo ========================================
pause
