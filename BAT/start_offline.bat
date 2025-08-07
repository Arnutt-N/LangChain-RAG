@echo off
echo ========================================
echo   RAG Chatbot - OFFLINE MODE
echo ========================================
echo.

REM Force offline mode
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set NO_INTERNET=1

REM Use pre-built vectors only
set USE_PREBUILT_ONLY=1

echo [*] Starting in OFFLINE mode...
echo [*] Using pre-built vectors only
echo [*] No internet connection required
echo.

cd /d D:\genAI\LangChain-RAG

REM Check if pre-built vectors exist
if not exist "prebuilt_vectors\faiss_index" (
    echo [ERROR] Pre-built vectors not found!
    echo Please run: python prebuild_vectors.py
    pause
    exit /b 1
)

echo [✓] Pre-built vectors found
echo [*] Starting server...
echo.

streamlit run streamlit_app.py --server.port 8501 --browser.gatherUsageStats false

pause
