"""
Quick test to verify the app is working
"""
import requests
import time

print("=" * 50)
print("Testing LangChain-RAG App")
print("=" * 50)
print()

# Check if app is running
url = "http://localhost:8501"

try:
    print(f"1. Checking if app is running at {url}...")
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        print("   ✅ App is running!")
        print(f"   Status code: {response.status_code}")
        
        # Check for common elements
        if "streamlit" in response.text.lower():
            print("   ✅ Streamlit framework detected")
        
        if "rag" in response.text.lower() or "chatbot" in response.text.lower():
            print("   ✅ RAG Chatbot UI detected")
            
    else:
        print(f"   ⚠️ App responded with status: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("   ❌ Cannot connect to app")
    print("   Please make sure the app is running:")
    print("   streamlit run streamlit_app.py")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print()
print("2. App Status Summary:")
print("   - URL: http://localhost:8501")
print("   - To stop: Press Ctrl+C in the terminal")
print("   - To access: Open browser and go to http://localhost:8501")
print()

# Check what's working
print("3. Component Status:")

# Check imports
components = {
    "LangChain": "import langchain",
    "Streamlit": "import streamlit",
    "FAISS": "import faiss",
    "Transformers": "from transformers import AutoModel",
    "Google Generative AI": "import google.generativeai",
}

for name, import_cmd in components.items():
    try:
        exec(import_cmd)
        print(f"   ✅ {name}: OK")
    except ImportError:
        print(f"   ❌ {name}: Not installed")

print()
print("=" * 50)
print("✨ App should be accessible at: http://localhost:8501")
print("=" * 50)
