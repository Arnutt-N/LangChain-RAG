"""
Simple test for embeddings - using a smaller model
"""

print("=" * 50)
print("Testing Simple Embeddings")
print("=" * 50)
print()

try:
    print("1. Testing sentence-transformers import...")
    from sentence_transformers import SentenceTransformer
    print("   ✅ Import successful")
    
    print("\n2. Loading small model (paraphrase-MiniLM-L3-v2)...")
    # Use smaller model for testing
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    print("   ✅ Model loaded")
    
    print("\n3. Testing encoding...")
    test_sentences = ["Hello world", "How are you?"]
    embeddings = model.encode(test_sentences)
    print(f"   ✅ Encoded {len(test_sentences)} sentences")
    print(f"   Embedding shape: {embeddings.shape}")
    
    print("\n4. Testing similarity...")
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"   ✅ Cosine similarity: {similarity:.3f}")
    
    print("\n✨ SUCCESS! Embeddings are working!")
    print("\nYou can now use this in your app.")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nTrying alternative method...")
    
    try:
        # Try using transformers directly
        print("\n5. Testing transformers library...")
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print("   ✅ Transformers method works!")
        
    except Exception as e2:
        print(f"   ❌ Transformers also failed: {e2}")
        print("\n⚠️ Please run: pip install sentence-transformers>=3.0.0")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease check your installation.")

print("\n" + "=" * 50)
