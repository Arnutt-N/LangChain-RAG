"""
Optimized Streamlit App Configuration
"""

import streamlit as st
import os
from pathlib import Path

# Performance optimizations
st.set_page_config(
    page_title="RAG Chatbot (Optimized)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed
)

# Disable Hugging Face symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Use local model cache
LOCAL_MODEL_PATH = "D:\\genAI\\LangChain-RAG\\models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = LOCAL_MODEL_PATH

# Optimize Hugging Face downloads
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"  # Allow online but use cache first

# Network optimization
os.environ["CURL_CA_BUNDLE"] = ""  # Disable SSL verification for faster downloads
os.environ["REQUESTS_CA_BUNDLE"] = ""

print("🚀 Optimizations loaded!")
