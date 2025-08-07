@echo off
echo ========================================
echo   Manual Model Download Instructions
echo ========================================
echo.
echo เนื่องจากอินเทอร์เน็ตไม่เสถียร แนะนำให้ดาวน์โหลด model ผ่าน browser:
echo.
echo 1. เปิด Browser ไปที่:
echo    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main
echo.
echo 2. Download ไฟล์ต่อไปนี้:
echo    - model.safetensors (90.9 MB)
echo    - config.json
echo    - tokenizer.json
echo    - tokenizer_config.json
echo.
echo 3. สร้างโฟลเดอร์:
echo    D:\genAI\LangChain-RAG\models\all-MiniLM-L6-v2\
echo.
echo 4. ใส่ไฟล์ที่ดาวน์โหลดลงในโฟลเดอร์นั้น
echo.
echo 5. รัน start_offline.bat
echo.
echo กด Enter เพื่อเปิด Browser...
pause

start https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main

echo.
echo กำลังสร้างโฟลเดอร์...
mkdir "D:\genAI\LangChain-RAG\models\all-MiniLM-L6-v2" 2>nul
echo โฟลเดอร์พร้อมแล้ว: D:\genAI\LangChain-RAG\models\all-MiniLM-L6-v2
echo.
echo หลังจากดาวน์โหลดเสร็จ ให้รัน start_offline.bat
pause
