import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.embedding import generate_embedding, EMBEDDING_ENABLED, get_embedding_dimension
from config.settings import settings # To ensure settings are loaded for logging level

logger = logging.getLogger(__name__)
# Configure logger if not already configured by a higher-level basicConfig
if not logger.hasHandlers():
    logging.basicConfig(level=settings.log_level.upper(), format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')


INDEX_DIR = os.path.abspath(os.path.dirname(__file__))
INDEX_FILE_PATH = os.path.join(INDEX_DIR, 'eido_schema_index.json')

class EidoSchemaRetriever:
    def __init__(self, index_path: str = INDEX_FILE_PATH):
        self.index_path = index_path
        self.index_data: Optional[Dict] = None
        self.chunk_texts: List[str] = []
        self.chunk_names: List[str] = [] # Names of the schema components/chunks
        self.embeddings: Optional[np.ndarray] = None
        self.is_ready = False
        self._load_index()

    def _load_index(self):
        if not EMBEDDING_ENABLED:
            logger.warning("Embedding service is disabled. EIDO Schema Retriever will not function.")
            self.is_ready = False
            return

        if not os.path.exists(self.index_path):
            logger.error(f"EIDO schema index file not found at: {self.index_path}")
            logger.error("Please run the RAG indexer script first: python utils/rag_indexer.py")
            self.is_ready = False
            return

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                self.index_data = json.load(f)

            if not self.index_data or "chunks" not in self.index_data or "embeddings" not in self.index_data:
                logger.error("Index file is invalid or missing required keys ('chunks', 'embeddings').")
                self.index_data = None; self.is_ready = False
                return

            # Chunks are expected to be a list of dicts like {"name": "ComponentName", "text": "Formatted text"}
            self.chunk_names = [chunk.get("name", f"UnnamedChunk_{i}") for i, chunk in enumerate(self.index_data["chunks"])]
            self.chunk_texts = [chunk.get("text", "") for chunk in self.index_data["chunks"]]
            self.embeddings = np.array(self.index_data["embeddings"])

            # Validate dimensions
            index_dim = self.index_data.get("embedding_dim", 0)
            current_model_dim = get_embedding_dimension()

            if index_dim != current_model_dim and current_model_dim != 0: # current_model_dim is 0 if embedding model failed to load
                logger.warning(f"INDEX MISMATCH: Index embedding dimension ({index_dim}) "
                               f"differs from current model's dimension ({current_model_dim}). "
                               f"Retrieval quality may be poor. PLEASE RE-INDEX with `python utils/rag_indexer.py`.")
                # Optionally, could prevent retriever from becoming ready:
                # self.is_ready = False; return

            if self.embeddings.shape[0] == 0 or (self.embeddings.shape[0] > 0 and self.embeddings.shape[1] != index_dim):
                 logger.error(f"Embeddings in index file have unexpected shape {self.embeddings.shape} or dimension mismatch with index_dim {index_dim}.")
                 self.is_ready = False; return
            
            if len(self.chunk_texts) != self.embeddings.shape[0]:
                logger.error("Mismatch between number of chunks and embeddings in index file.")
                self.is_ready = False; return

            self.is_ready = True
            logger.info(f"EIDO Schema Retriever initialized successfully with {len(self.chunk_texts)} chunks from '{self.index_path}'.")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON index file {self.index_path}: {e}")
            self.is_ready = False
        except Exception as e:
            logger.error(f"Error loading EIDO schema index: {e}", exc_info=True)
            self.is_ready = False

    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        if not self.is_ready:
            logger.warning("Retriever is not ready. Cannot retrieve context. Check logs for initialization errors (e.g., index file missing or embedding model mismatch).")
            return []
        if not EMBEDDING_ENABLED: # Double check, though is_ready should catch this.
            logger.warning("Embeddings are disabled. Cannot retrieve context.")
            return []
        if not query or not isinstance(query, str):
            logger.warning(f"Invalid or empty query for retrieval: {query}")
            return []
        if self.embeddings is None or self.embeddings.shape[0] == 0:
             logger.error("No embeddings loaded in retriever. Cannot retrieve context.")
             return []

        try:
            query_embedding = generate_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate embedding for query, cannot retrieve context.")
                return []

            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            # Check for dimension mismatch here too, as a safeguard
            if query_embedding_np.shape[1] != self.embeddings.shape[1]:
                logger.error(f"Query embedding dimension ({query_embedding_np.shape[1]}) "
                               f"mismatches index embedding dimension ({self.embeddings.shape[1]}). "
                               f"Cannot compute similarity. Ensure embedding models are consistent and re-index if necessary.")
                return []

            similarities = cosine_similarity(query_embedding_np, self.embeddings)[0]
            num_chunks = len(self.chunk_texts)
            k = min(top_k, num_chunks)
            if k <= 0: return []

            # Get indices of top k scores (argsort sorts ascending, so take last k and reverse)
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            retrieved_chunks_details = [] # For logging
            retrieved_chunk_texts = []
            for rank, i in enumerate(top_k_indices):
                score = similarities[i]
                name = self.chunk_names[i]
                text = self.chunk_texts[i]
                retrieved_chunks_details.append({"rank": rank + 1, "score": f"{score:.4f}", "name": name})
                retrieved_chunk_texts.append(text)
            
            logger.info(f"Retrieved {len(retrieved_chunk_texts)} context chunks for query '{query[:50]}...'. Top results: {retrieved_chunks_details}")
            return retrieved_chunk_texts

        except Exception as e:
            logger.error(f"Error during context retrieval for query '{query[:50]}...': {e}", exc_info=True)
            return []

eido_retriever = EidoSchemaRetriever()

if __name__ == "__main__":
    if eido_retriever.is_ready:
        test_queries = [
            "information about location component structure and required fields like coordinates or address",
            "how to represent an agency that reported an incident",
            "details for NotesType and its properties",
            "what is EmergencyIncidentDataObjectType"
        ]
        for tq in test_queries:
            print(f"\n--- Testing retrieval for query: '{tq}' ---")
            context = eido_retriever.retrieve_context(tq, top_k=2)
            if context:
                for i, chunk_text in enumerate(context):
                    print(f"--- Chunk {i+1} ({eido_retriever.chunk_names[np.where(np.array(eido_retriever.chunk_texts) == chunk_text)[0][0]] if chunk_text in eido_retriever.chunk_texts else 'N/A'}) ---")
                    print(chunk_text[:300] + "...") # Print snippet
                    print("-" * 20)
            else:
                print("No context retrieved or retriever not ready.")
    else:
        print("\nRetriever not ready. Run `python utils/rag_indexer.py` first or check logs for errors.")