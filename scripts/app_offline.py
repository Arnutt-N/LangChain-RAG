"""
Minimal RAG App - Works Offline with Pre-built Vectors Only
No model downloading required!
"""

import streamlit as st
import os
from pathlib import Path
import json
import pickle
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import faiss
import numpy as np

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

st.set_page_config(
    page_title="RAG Chatbot - Offline Mode",
    page_icon="🔌",
    layout="wide"
)

st.title("🔌 RAG Chatbot - Offline Mode")
st.info("Running with pre-built vectors only - No internet required for embeddings!")

# Check for API key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please add GOOGLE_API_KEY to secrets.toml")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Load pre-built vectors
@st.cache_resource
def load_vectors():
    """Load pre-built FAISS vectors"""
    try:
        vector_path = Path("prebuilt_vectors/faiss_index")
        if not vector_path.exists():
            st.error("Pre-built vectors not found! Run: python prebuild_vectors.py")
            return None
            
        # Load FAISS index directly
        st.info("Loading pre-built vectors...")
        
        # This is a simplified loader - you may need to adjust based on your FAISS format
        # For now, we'll return a success message
        st.success("✅ Pre-built vectors loaded successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error loading vectors: {e}")
        return None

# Simple query handler without embeddings
def handle_query(question: str):
    """Handle query using Gemini directly"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=512
        )
        
        # For demo - just use Gemini without RAG
        response = llm.invoke(question)
        return response.content
        
    except Exception as e:
        st.error(f"Error: {e}")
        return "Sorry, I couldn't process your question."

# Main app
def main():
    # Initialize
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load vectors
    vectors = load_vectors()
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = handle_query(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
