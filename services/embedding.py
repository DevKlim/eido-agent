# services/embedding.py
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import logging
import numpy as np

# Import settings
from config.settings import settings

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level.upper())

# --- Embedding Model Setup ---
EMBEDDING_DIM = 0
embedding_model = None
MODEL_NAME = settings.embedding_model_name

try:
    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    # You might want to specify a cache directory for models
    # embedding_model = SentenceTransformer(MODEL_NAME, cache_folder='./model_cache')
    embedding_model = SentenceTransformer(MODEL_NAME)
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    logger.info(f"Embedding model '{MODEL_NAME}' loaded successfully (Dimension: {EMBEDDING_DIM}).")
    EMBEDDING_ENABLED = True
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model '{MODEL_NAME}': {e}", exc_info=True)
    logger.error("Text embedding generation will be disabled.")
    EMBEDDING_ENABLED = False
    embedding_model = None
    EMBEDDING_DIM = 0


def generate_embedding(text: Optional[str]) -> Optional[List[float]]:
    """
    Generates a sentence embedding for the given text using the configured model.
    Returns a list of floats or None if embedding is disabled, text is invalid, or an error occurs.
    """
    if not EMBEDDING_ENABLED or not text or not isinstance(text, str) or not text.strip():
        if not EMBEDDING_ENABLED:
            logger.debug("Embedding generation skipped (service disabled).")
        return None

    try:
        logger.debug(f"Generating embedding for text snippet: '{text[:50]}...'")
        # Encode the text. Convert to numpy first for potential operations, then to list for storage/JSON.
        embedding_vector = embedding_model.encode(text.strip(), convert_to_numpy=True)
        # Optional: Normalize the embedding vector (improves cosine similarity results sometimes)
        # embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
        result = embedding_vector.tolist()
        logger.debug(f"Embedding generated successfully (Dim: {len(result)}).")
        return result
    except Exception as e:
        logger.error(f"Error generating embedding for text: {e}", exc_info=True)
        return None

def get_embedding_dimension() -> int:
     """Returns the dimension of the loaded embedding model."""
     return EMBEDDING_DIM