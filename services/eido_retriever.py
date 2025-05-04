# services/eido_retriever.py
import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Ensure services are importable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.embedding import generate_embedding, EMBEDDING_ENABLED

logger = logging.getLogger(__name__)

# Define path relative to project root
INDEX_DIR = os.path.abspath(os.path.dirname(__file__)) # Index is in services dir
INDEX_FILE_PATH = os.path.join(INDEX_DIR, 'eido_schema_index.json')

class EidoSchemaRetriever:
    """Retrieves relevant EIDO schema chunks based on semantic similarity."""

    def __init__(self, index_path: str = INDEX_FILE_PATH):
        self.index_path = index_path
        self.index_data: Optional[Dict] = None
        self.chunk_texts: List[str] = []
        self.chunk_names: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_ready = False
        self._load_index()

    def _load_index(self):
        """Loads the pre-computed index from the JSON file."""
        if not EMBEDDING_ENABLED:
            logger.warning("Embedding service is disabled. Retriever will not function.")
            return

        if not os.path.exists(self.index_path):
            logger.error(f"EIDO schema index file not found at: {self.index_path}")
            logger.error("Please run the indexer script first: python utils/rag_indexer.py")
            return

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                self.index_data = json.load(f)

            if not self.index_data or "chunks" not in self.index_data or "embeddings" not in self.index_data:
                logger.error("Index file is invalid or missing required keys ('chunks', 'embeddings').")
                self.index_data = None
                return

            self.chunk_names = [chunk.get("name", "Unknown") for chunk in self.index_data["chunks"]]
            self.chunk_texts = [chunk.get("text", "") for chunk in self.index_data["chunks"]]
            self.embeddings = np.array(self.index_data["embeddings"])

            # Validate dimensions
            expected_dim = self.index_data.get("embedding_dim", 0)
            if self.embeddings.shape[1] != expected_dim:
                 logger.warning(f"Index embedding dimension ({self.embeddings.shape[1]}) doesn't match expected ({expected_dim}).")
            if len(self.chunk_texts) != self.embeddings.shape[0]:
                logger.error("Mismatch between number of chunks and embeddings in index file.")
                self.is_ready = False
                return

            self.is_ready = True
            logger.info(f"EIDO Schema Retriever initialized successfully with {len(self.chunk_texts)} chunks.")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON index file {self.index_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading EIDO schema index: {e}", exc_info=True)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves the top_k most relevant schema chunk texts for a given query.
        """
        if not self.is_ready or not EMBEDDING_ENABLED:
            logger.warning("Retriever is not ready or embeddings are disabled. Cannot retrieve context.")
            return []
        if not query:
            logger.warning("Empty query received for retrieval.")
            return []
        if self.embeddings is None or len(self.embeddings) == 0:
             logger.error("No embeddings loaded in retriever.")
             return []


        try:
            query_embedding = generate_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate embedding for query.")
                return []

            query_embedding_np = np.array(query_embedding).reshape(1, -1) # Reshape for cosine_similarity

            # Calculate similarities
            similarities = cosine_similarity(query_embedding_np, self.embeddings)[0] # Get the single row of similarities

            # Get top_k indices (handling cases where k > number of chunks)
            num_chunks = len(self.chunk_texts)
            k = min(top_k, num_chunks)
            if k <= 0: return []

            # Get indices of top k scores using argsort
            # Argsort sorts in ascending order, so we take the last k indices for descending order
            top_k_indices = np.argsort(similarities)[-k:][::-1] # Get top k indices, then reverse to have highest first

            # Retrieve corresponding chunks and log info
            retrieved_chunks = []
            logger.debug(f"Retrieval results for query '{query[:50]}...':")
            for i in top_k_indices:
                score = similarities[i]
                name = self.chunk_names[i]
                text = self.chunk_texts[i]
                logger.debug(f"  - Rank {len(retrieved_chunks)+1}: Score={score:.4f}, Component='{name}'")
                retrieved_chunks.append(text)

            logger.info(f"Retrieved {len(retrieved_chunks)} context chunks for query.")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error during context retrieval: {e}", exc_info=True)
            return []

# --- Singleton Instance ---
# Initialize retriever when module is loaded
eido_retriever = EidoSchemaRetriever()

# Example usage (optional)
if __name__ == "__main__":
    if eido_retriever.is_ready:
        test_query = "details about location component structure and required fields"
        print(f"\nTesting retrieval for query: '{test_query}'")
        context = eido_retriever.retrieve_context(test_query, top_k=2)
        if context:
            print("\n--- Retrieved Context ---")
            for i, chunk in enumerate(context):
                print(f"--- Chunk {i+1} ---")
                print(chunk)
                print("-" * 20)
        else:
            print("No context retrieved.")
    else:
        print("\nRetriever not ready. Run utils/rag_indexer.py first.")