from sentence_transformers import SentenceTransformer
from typing import Optional, List
import logging
import os
from threading import Lock

from config.settings import settings

logger = logging.getLogger(__name__)

# --- LAZY LOADING SETUP ---
# Do NOT load the model at import time. Initialize to None.
embedding_model: Optional[SentenceTransformer] = None
MODEL_NAME: str = settings.embedding_model_name
EMBEDDING_DIM: Optional[int] = None
EMBEDDING_ENABLED: bool = True  # Assume enabled unless loading fails
model_lock = Lock() # To prevent race conditions if multiple requests come at once

def _get_model() -> Optional[SentenceTransformer]:
    """
    Lazily loads the embedding model using a singleton pattern.
    This function is thread-safe.
    """
    global embedding_model
    global EMBEDDING_ENABLED
    global EMBEDDING_DIM

    # First check without a lock for performance
    if embedding_model is not None:
        return embedding_model

    with model_lock:
        # Double-check inside the lock to ensure it wasn't loaded by another thread
        if embedding_model is not None:
            return embedding_model

        if not EMBEDDING_ENABLED: # If loading previously failed
            return None

        try:
            logger.info(f"LAZY LOADING: Attempting to load embedding model for the first time: {MODEL_NAME}...")
            model = SentenceTransformer(MODEL_NAME)
            
            dim_candidate = model.get_sentence_embedding_dimension()
            if dim_candidate and isinstance(dim_candidate, int):
                EMBEDDING_DIM = dim_candidate
                embedding_model = model
                logger.info(f"Embedding model '{MODEL_NAME}' loaded successfully (Dimension: {EMBEDDING_DIM}).")
                return embedding_model
            else:
                logger.error(f"Failed to get valid dimension for model '{MODEL_NAME}'. Embeddings will be disabled.")
                EMBEDDING_ENABLED = False
                return None

        except Exception as e:
            logger.error(f"CRITICAL: Failed to lazy-load SentenceTransformer model '{MODEL_NAME}': {e}", exc_info=True)
            logger.error("Text embedding generation will be DISABLED for the lifetime of this process.")
            EMBEDDING_ENABLED = False
            return None

def generate_embedding(text: Optional[str]) -> Optional[List[float]]:
    """Generates an embedding for text, loading the model on first use."""
    
    # Get the model. This will trigger the lazy load on the first call.
    model = _get_model()
    
    if model is None:
        return None
    if not text or not isinstance(text, str) or not text.strip():
        return None

    try:
        embedding_vector = model.encode(text.strip(), convert_to_numpy=True)
        return embedding_vector.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text[:50]}...': {e}", exc_info=True)
        return None

def get_embedding_dimension() -> int:
    """Returns the dimension of the embedding model, loading it if necessary."""
    # Ensure the model is loaded to know its dimension
    _get_model()
    return EMBEDDING_DIM if EMBEDDING_DIM is not None else 0

# Self-test when module is run directly
if __name__ == "__main__":
    print(f"--- Embedding Service Self-Test (Lazy Loading) ---")
    print(f"Model Name: {MODEL_NAME}")
    
    test_sentence = "This is a test sentence for the lazy-loaded embedding service."
    print(f"\nTesting with sentence: \"{test_sentence}\"")
    start_time =  __import__('time').time()
    emb = generate_embedding(test_sentence)
    duration = __import__('time').time() - start_time
    
    if emb:
        print(f"Successfully generated embedding in {duration:.2f}s.")
        print(f"Generated embedding (first 5 values): {emb[:5]}...")
        print(f"Full embedding length: {len(emb)}")
        if len(emb) == get_embedding_dimension():
            print("Embedding dimension matches expected. Test PASSED.")
        else:
            print(f"Dimension MISMATCH! Expected {get_embedding_dimension()}, got {len(emb)}. Test FAILED.")
    else:
        print("Failed to generate embedding for test sentence. Test FAILED.")

    print(f"\n--- Self-Test Finished ---")