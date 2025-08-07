"""
Test and diagnose embeddings issues
"""

import sys
import os

print("=" * 50)
print("Embeddings Diagnostic Tool")
print("=" * 50)
print()

# Check Python version
print(f"Python version: {sys.version}")
print()

# Test imports
tests = {
    "huggingface_hub": ["from huggingface_hub import hf_hub_download", "from huggingface_hub import cached_download"],
    "langchain_huggingface": ["from langchain_huggingface import HuggingFaceEmbeddings"],
    "langchain_community": ["from langchain_community.embeddings import HuggingFaceEmbeddings"],
    "sentence_transformers": ["from sentence_transformers import SentenceTransformer"],
}

print("Testing imports:")
for package, imports in tests.items():
    print(f"\n{package}:")
    for imp in imports:
        try:
            exec(imp)
            print(f"  ✅ {imp}")
        except Exception as e:
            print(f"  ❌ {imp}")
            print(f"     Error: {e}")

print("\n" + "=" * 50)
print("Attempting to create working embeddings...")
print("=" * 50)

# Try to create embeddings
success = False

# Method 1: New langchain-huggingface
try:
    print("\n1. Trying langchain-huggingface...")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    test = embeddings.embed_query("test")
    print(f"   ✅ Success! Embedding dimension: {len(test)}")
    success = True
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Method 2: Direct sentence-transformers
if not success:
    try:
        print("\n2. Trying sentence-transformers directly...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test = model.encode(["test"])[0]
        print(f"   ✅ Success! Embedding dimension: {len(test)}")
        success = True
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print("\n" + "=" * 50)
if success:
    print("✅ EMBEDDINGS WORKING!")
    print("\nYou can now run: streamlit run streamlit_app.py")
else:
    print("❌ NO WORKING EMBEDDINGS FOUND")
    print("\nPlease run: fix_huggingface.bat")
print("=" * 50)
