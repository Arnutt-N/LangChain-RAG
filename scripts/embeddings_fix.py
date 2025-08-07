"""
Updated streamlit_app.py imports to use sentence-transformers
instead of deprecated HuggingFaceEmbeddings
"""

# At the top of streamlit_app.py, replace the embeddings import section:

@st.cache_resource
def get_embeddings(use_qwen: bool = True):
    """Initialize embeddings - Qwen if HF_TOKEN available, else SentenceTransformer"""
    
    if use_qwen and HF_TOKEN:
        try:
            embeddings = QwenEmbeddings(
                api_key=HF_TOKEN,
                model_name="Qwen/Qwen3-Embedding-8B",
                provider="nebius",
                batch_size=16,
                max_retries=3
            )
            st.session_state.embeddings_type = "qwen"
            return embeddings
        except Exception as e:
            st.warning(f"Failed to initialize Qwen embeddings: {e}")
            st.info("Falling back to Sentence Transformers...")
    
    # Use Sentence Transformers instead of deprecated HuggingFaceEmbeddings
    try:
        from sentence_transformers import SentenceTransformer
        from langchain.embeddings.base import Embeddings
        
        class SentenceTransformerEmbeddings(Embeddings):
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                cache_dir = Path("D:/genAI/LangChain-RAG/models")
                cache_dir.mkdir(exist_ok=True)
                self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
            
            def embed_documents(self, texts):
                return self.model.encode(texts, convert_to_numpy=True).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text], convert_to_numpy=True)[0].tolist()
        
        embeddings = SentenceTransformerEmbeddings()
        st.session_state.embeddings_type = "sentence-transformers"
        return embeddings
        
    except Exception as e:
        st.error(f"Failed to initialize embeddings: {e}")
        return None
