@echo off
echo ========================================
echo   Fix Python Dependencies Conflicts
echo ========================================
echo.

cd /d D:\genAI\LangChain-RAG

echo [1/5] Backing up current requirements...
copy requirements.txt requirements_backup.txt >nul 2>&1
echo Done.
echo.

echo [2/5] Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel
echo.

echo [3/5] Fixing dependency conflicts...
echo.

REM Fix specific conflicts
echo Updating huggingface-hub...
pip install huggingface-hub==0.34.0 --upgrade
echo.

echo Updating fastapi and uvicorn...
pip install fastapi==0.115.0 uvicorn==0.34.0 --upgrade
echo.

echo [4/5] Installing fixed requirements...
pip install -r requirements_fixed.txt --upgrade
echo.

echo [5/5] Verifying installation...
python -c "import langchain; import streamlit; import faiss; print('✅ Core packages OK')"
python -c "from sentence_transformers import SentenceTransformer; print('✅ Sentence Transformers OK')"
python -c "import google.generativeai; print('✅ Google Generative AI OK')"
echo.

echo ========================================
echo   Dependency fixes completed!
echo ========================================
echo.
echo You can now run:
echo   streamlit run streamlit_app.py
echo.
pause
