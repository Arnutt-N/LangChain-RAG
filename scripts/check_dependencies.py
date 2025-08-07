"""
Quick dependency check and fix script
"""

import subprocess
import sys

def run_command(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def main():
    print("=" * 50)
    print("Dependency Conflict Checker & Fixer")
    print("=" * 50)
    print()
    
    # Check current conflicts
    print("1. Checking for conflicts...")
    stdout, stderr, code = run_command("pip check")
    
    if code == 0 and not stderr:
        print("✅ No conflicts found!")
    else:
        print("⚠️ Conflicts detected:")
        print(stdout)
        print(stderr)
        print()
        
        # Auto fix
        print("2. Attempting automatic fix...")
        
        fixes = [
            ("huggingface-hub", "0.34.0"),
            ("fastapi", "0.115.0"),
            ("uvicorn", "0.34.0"),
            ("sentence-transformers", "2.2.2"),
        ]
        
        for package, version in fixes:
            print(f"   Installing {package}=={version}...")
            cmd = f"pip install {package}=={version} --upgrade --quiet"
            stdout, stderr, code = run_command(cmd)
            if code == 0:
                print(f"   ✅ {package} updated")
            else:
                print(f"   ❌ Failed to update {package}")
                print(f"      Error: {stderr}")
        
        print()
        print("3. Verifying fixes...")
        stdout, stderr, code = run_command("pip check")
        
        if code == 0 and not stderr:
            print("✅ All conflicts resolved!")
        else:
            print("⚠️ Some conflicts remain:")
            print(stdout)
            print()
            print("Try running: pip install -r requirements_fixed.txt --upgrade")
    
    print()
    print("4. Testing imports...")
    
    test_imports = [
        "import streamlit",
        "import langchain",
        "import faiss",
        "from sentence_transformers import SentenceTransformer",
        "import google.generativeai",
    ]
    
    all_ok = True
    for imp in test_imports:
        try:
            exec(imp)
            print(f"   ✅ {imp}")
        except ImportError as e:
            print(f"   ❌ {imp} - {e}")
            all_ok = False
    
    print()
    if all_ok:
        print("✨ All dependencies OK! You can run the app now.")
    else:
        print("⚠️ Some imports failed. Run: fix_dependencies.bat")
    
    print()
    print("=" * 50)

if __name__ == "__main__":
    main()
