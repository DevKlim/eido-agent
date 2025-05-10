from sentence_transformers import SentenceTransformer
from typing import Optional, List
import logging
import numpy as np
import os

from config.settings import settings # For log level and model name

logger = logging.getLogger(__name__)
# Configure logger if not already configured by a higher-level basicConfig
if not logger.hasHandlers(): # Avoid adding multiple handlers if basicConfig was called elsewhere
    logging.basicConfig(level=settings.log_level.upper(), format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

EMBEDDING_DIM: int = 0
embedding_model: Optional[SentenceTransformer] = None
MODEL_NAME: str = settings.embedding_model_name
EMBEDDING_ENABLED: bool = False

# Define a cache directory for sentence-transformers models
# This path should be relative to the project root or an absolute path
# For simplicity, let's try to put it inside the project structure if writable.
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, '.model_cache') # Hidden cache dir

try:
    # Ensure cache directory exists if you choose to use one
    # os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    # logger.info(f"Using model cache directory: {MODEL_CACHE_DIR}")
    
    logger.info(f"Attempting to load embedding model: {MODEL_NAME}...")
    # embedding_model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)
    embedding_model = SentenceTransformer(MODEL_NAME) # Default cache path is fine too
    
    if embedding_model:
        dim_candidate = embedding_model.get_sentence_embedding_dimension()
        if dim_candidate is not None and isinstance(dim_candidate, int):
            EMBEDDING_DIM = dim_candidate
            EMBEDDING_ENABLED = True
            logger.info(f"Embedding model '{MODEL_NAME}' loaded successfully (Dimension: {EMBEDDING_DIM}). Embeddings ENABLED.")
        else:
            logger.error(f"Failed to get valid dimension for model '{MODEL_NAME}'. Got: {dim_candidate}. Embeddings DISABLED.")
            embedding_model = None # Explicitly set to None
    else: # Should not happen if SentenceTransformer call was successful without error
        logger.error(f"SentenceTransformer returned None for model '{MODEL_NAME}'. Embeddings DISABLED.")

except Exception as e:
    logger.error(f"CRITICAL: Failed to load SentenceTransformer model '{MODEL_NAME}': {e}", exc_info=True)
    logger.error("Text embedding generation will be DISABLED. RAG features will not work.")
    EMBEDDING_ENABLED = False
    embedding_model = None
    EMBEDDING_DIM = 0


def generate_embedding(text: Optional[str]) -> Optional[List[float]]:
    if not EMBEDDING_ENABLED or embedding_model is None:
        # logger.debug("Embedding generation skipped (service disabled or model not loaded).")
        return None
    if not text or not isinstance(text, str) or not text.strip():
        # logger.debug(f"Embedding generation skipped for invalid/empty text: '{str(text)[:50]}...'")
        return None

    try:
        # logger.debug(f"Generating embedding for text snippet: '{text[:50]}...'")
        embedding_vector = embedding_model.encode(text.strip(), convert_to_numpy=True)
        # Optional: Normalize for cosine similarity, though many models are pre-normalized
        # embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
        result = embedding_vector.tolist()
        # logger.debug(f"Embedding generated successfully (Dim: {len(result)}).")
        return result
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text[:50]}...': {e}", exc_info=True)
        return None

def get_embedding_dimension() -> int:
     return EMBEDDING_DIM

# Self-test when module is run directly
if __name__ == "__main__":
    print(f"--- Embedding Service Self-Test ---")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Embedding Enabled: {EMBEDDING_ENABLED}")
    print(f"Embedding Dimension: {EMBEDDING_DIM}")
    if EMBEDDING_ENABLED and embedding_model:
        test_sentence = "This is a test sentence for the EIDO Sentinel embedding service."
        print(f"\nTesting with sentence: \"{test_sentence}\"")
        emb = generate_embedding(test_sentence)
        if emb:
            print(f"Generated embedding (first 5 values): {emb[:5]}...")
            print(f"Full embedding length: {len(emb)}")
            if len(emb) == EMBEDDING_DIM:
                print("Embedding dimension matches expected. Test PASSED.")
            else:
                print(f"Dimension MISMATCH! Expected {EMBEDDING_DIM}, got {len(emb)}. Test FAILED.")
        else:
            print("Failed to generate embedding for test sentence. Test FAILED.")
        
        print("\nTesting with empty string:")
        emb_empty = generate_embedding("")
        if emb_empty is None:
            print("Correctly returned None for empty string. Test PASSED.")
        else:
            print(f"Incorrectly returned embedding for empty string: {emb_empty}. Test FAILED.")

    else:
        print("\nEmbedding model not loaded. Cannot run functional tests.")
    print(f"--- Self-Test Finished ---")