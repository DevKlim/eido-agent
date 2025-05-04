# utils/rag_indexer.py
import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Any

# Ensure services and utils are importable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.schema_parser import load_openapi_schema, format_component_details_for_llm
from services.embedding import generate_embedding, get_embedding_dimension, EMBEDDING_ENABLED

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# Define paths relative to project root
SCHEMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EIDO-JSON', 'Schema', 'openapi.yaml'))
INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'services')) # Save index in services dir
INDEX_FILE_PATH = os.path.join(INDEX_DIR, 'eido_schema_index.json')

def create_schema_chunks(schema: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Creates text chunks from schema components."""
    chunks = []
    if not schema or 'components' not in schema or 'schemas' not in schema['components']:
        logger.error("Cannot create chunks: Invalid schema structure.")
        return chunks

    component_schemas = schema['components']['schemas']
    for component_name in component_schemas.keys():
        # Skip simple types or irrelevant base types if needed
        # if "SimpleType" in component_name or component_name in ["ProblemDetails"]:
        #     continue

        chunk_text = format_component_details_for_llm(schema, component_name)
        if chunk_text:
            # Store tuple: (component_name, formatted_text_chunk)
            chunks.append((component_name, chunk_text))
        else:
            logger.warning(f"Could not format details for component: {component_name}")

    logger.info(f"Created {len(chunks)} text chunks from schema components.")
    return chunks

def build_and_save_index(chunks: List[Tuple[str, str]], output_path: str = INDEX_FILE_PATH):
    """Generates embeddings and saves the index."""
    if not EMBEDDING_ENABLED:
        logger.error("Embedding service is disabled. Cannot build index.")
        return False
    if not chunks:
        logger.error("No chunks provided to build index.")
        return False

    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = []
    chunk_data = [] # Store component name and text

    for i, (name, text) in enumerate(chunks):
        logger.debug(f"Embedding chunk {i+1}/{len(chunks)}: {name}")
        embedding = generate_embedding(text)
        if embedding:
            embeddings.append(embedding)
            chunk_data.append({"name": name, "text": text})
        else:
            logger.warning(f"Failed to generate embedding for chunk: {name}. Skipping.")

    if not embeddings:
        logger.error("No embeddings were generated. Index not saved.")
        return False

    # Convert embeddings to numpy array for potential later use, but save as list
    embeddings_list = np.array(embeddings).tolist()
    embedding_dim = get_embedding_dimension()

    index_data = {
        "embedding_dim": embedding_dim,
        "chunks": chunk_data, # List of {"name": str, "text": str}
        "embeddings": embeddings_list # List of lists (embeddings)
    }

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"Successfully saved EIDO schema index to: {output_path}")
        return True
    except IOError as e:
        logger.error(f"Error saving index file to {output_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving index: {e}", exc_info=True)
        return False

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting EIDO Schema Indexing ---")
    eido_schema = load_openapi_schema()
    if eido_schema:
        schema_chunks = create_schema_chunks(eido_schema)
        if schema_chunks:
            build_and_save_index(schema_chunks)
        else:
            logger.error("Failed to create schema chunks.")
    else:
        logger.error("Failed to load EIDO schema. Indexing aborted.")
    logger.info("--- Indexing Finished ---")