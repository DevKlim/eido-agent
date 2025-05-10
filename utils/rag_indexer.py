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
from config.settings import settings # For logging level

# --- Logging Setup ---
# Configure logger for this script
log_level_script = settings.log_level.upper() if hasattr(settings, 'log_level') else 'INFO'
logging.basicConfig(level=log_level_script, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("RAGIndexer")
logger.info(f"RAG Indexer started with log level {log_level_script}")

# Define paths relative to project root
# Assuming this script is in utils/
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(UTILS_DIR, '..'))
SCHEMA_PATH = os.path.join(PROJECT_ROOT, 'EIDO-JSON', 'Schema', 'openapi.yaml')
INDEX_DIR = os.path.join(PROJECT_ROOT, 'services') # Save index in services dir
INDEX_FILE_PATH = os.path.join(INDEX_DIR, 'eido_schema_index.json')

def create_schema_chunks(schema: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Creates text chunks from schema components suitable for RAG."""
    chunks = []
    if not schema or 'components' not in schema or 'schemas' not in schema['components']:
        logger.error("Cannot create chunks: Invalid OpenAPI schema structure (missing components/schemas).")
        return chunks

    component_schemas = schema['components']['schemas']
    logger.info(f"Found {len(component_schemas)} components in schema.")

    skipped_count = 0
    for component_name, component_data in component_schemas.items():
        # Optionally skip overly simple or generic types if they don't add much value
        # Example: skip primitive type aliases if they exist
        # if component_data.get('type') in ['string', 'integer', 'boolean', 'number'] and not component_data.get('properties'):
        #      logger.debug(f"Skipping simple type component: {component_name}")
        #      skipped_count += 1
        #      continue

        chunk_text = format_component_details_for_llm(schema, component_name)
        if chunk_text:
            chunks.append((component_name, chunk_text)) # Tuple: (ComponentName, FormattedText)
        else:
            logger.warning(f"Could not format details for component: {component_name}")
            skipped_count += 1

    logger.info(f"Created {len(chunks)} text chunks from schema components. Skipped {skipped_count}.")
    return chunks

def build_and_save_index(chunks: List[Tuple[str, str]], output_path: str = INDEX_FILE_PATH):
    """Generates embeddings for schema chunks and saves the index as JSON."""
    if not EMBEDDING_ENABLED:
        logger.error("Embedding service is DISABLED. Cannot build RAG index. Check services/embedding.py logs.")
        return False
    if not chunks:
        logger.error("No schema chunks provided to build index.")
        return False

    embedding_dim = get_embedding_dimension()
    if embedding_dim == 0:
         logger.error("Embedding dimension is 0. Cannot generate valid embeddings. Check embedding model loading.")
         return False

    logger.info(f"Generating embeddings for {len(chunks)} chunks (Dimension: {embedding_dim})...")
    embeddings_list = []
    chunk_data_list = [] # Store {"name": ComponentName, "text": FormattedText}

    for i, (name, text) in enumerate(chunks):
        logger.debug(f"Embedding chunk {i+1}/{len(chunks)}: '{name}'")
        embedding = generate_embedding(text)
        if embedding and len(embedding) == embedding_dim: # Ensure correct dimension
            embeddings_list.append(embedding)
            chunk_data_list.append({"name": name, "text": text})
        elif embedding: # Log dimension mismatch
             logger.warning(f"Failed to generate embedding for chunk '{name}' with correct dimension ({len(embedding)} vs {embedding_dim}). Skipping.")
        else: # Log failure
            logger.warning(f"Failed to generate embedding for chunk '{name}'. Skipping.")

    if not embeddings_list:
        logger.error("No embeddings were successfully generated. Index not saved.")
        return False

    logger.info(f"Generated {len(embeddings_list)} valid embeddings.")

    # Prepare index data for JSON serialization (embeddings as list of lists)
    index_data_to_save = {
        "embedding_model": settings.embedding_model_name, # Store which model was used
        "embedding_dim": embedding_dim,
        "chunks": chunk_data_list,
        "embeddings": embeddings_list
    }

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data_to_save, f, indent=2) # Use indent for readability
        logger.info(f"Successfully saved EIDO schema RAG index ({len(chunk_data_list)} chunks) to: {output_path}")
        return True
    except IOError as e:
        logger.error(f"Error saving index file to {output_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving RAG index: {e}", exc_info=True)
        return False

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting EIDO Schema RAG Indexing ---")
    eido_schema = load_openapi_schema(SCHEMA_PATH)
    if eido_schema:
        schema_chunks = create_schema_chunks(eido_schema)
        if schema_chunks:
            if build_and_save_index(schema_chunks, INDEX_FILE_PATH):
                 logger.info("Indexing completed successfully.")
            else:
                 logger.error("Index building failed.")
        else:
            logger.error("Failed to create schema chunks. Indexing aborted.")
    else:
        logger.error(f"Failed to load EIDO schema from {SCHEMA_PATH}. Indexing aborted.")
    logger.info("--- RAG Indexing Finished ---")