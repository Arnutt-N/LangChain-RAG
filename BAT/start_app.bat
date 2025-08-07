@echo off
echo ========================================
echo   RAG Chatbot - Quick Start (Optimized)
echo ========================================
echo.

REM Set environment variables for optimization
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set SENTENCE_TRANSFORMERS_HOME=D:\genAI\LangChain-RAG\models
set HF_HUB_OFFLINE=1
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_HEADLESS=true

echo [1/3] Setting up optimizations...
echo.

REM Check if model exists
if not exist "D:\genAI\LangChain-RAG\models\sentence-transformers_all-MiniLM-L6-v2" (
    echo [!] Model not found. Downloading first time...
    python D:\genAI\LangChain-RAG\download_model.py
    if errorlevel 1 (
        echo [ERROR] Failed to download model
        pause
        exit /b 1
    )
)

echo [2/3] Model ready. Starting server...
echo.

REM Start streamlit with optimizations
echo [3/3] Launching app at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d D:\genAI\LangChain-RAG
streamlit run streamlit_app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false

pause
