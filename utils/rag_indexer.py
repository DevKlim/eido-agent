import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Any
import xml.etree.ElementTree as ET

# Ensure services and utils are importable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import settings first to configure logging level
from config.settings import settings
from utils.schema_parser import load_openapi_schema, format_component_details_for_llm
from services.embedding import generate_embedding, get_embedding_dimension, EMBEDDING_ENABLED


# --- Logging Setup ---
# Configure logger for this script
log_level_script = settings.log_level.upper() if hasattr(settings, 'log_level') else 'INFO'
logging.basicConfig(level=log_level_script, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("RAGIndexer")
logger.info(f"RAG Indexer started with log level {log_level_script}")

# Define paths relative to project root
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(UTILS_DIR, '..'))
SCHEMA_PATH = os.path.join(PROJECT_ROOT, 'schema', 'openapi.yaml')
XML_CONTEXT_PATH = os.path.join(PROJECT_ROOT, 'eido_templates', 'EIDOContext.xml')
INDEX_DIR = os.path.join(PROJECT_ROOT, 'services')
INDEX_FILE_PATH = os.path.join(INDEX_DIR, 'eido_schema_index.json')

def create_xml_context_chunks(xml_path: str) -> List[Tuple[str, str]]:
    """Parses the EIDOContext.xml file and creates text chunks."""
    chunks = []
    if not os.path.exists(xml_path):
        logger.error(f"XML context file not found at: {xml_path}")
        return chunks
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract from Summary
        summary = root.find('Summary')
        if summary is not None and summary.text:
            chunks.append(('EIDO Summary', f"Overall EIDO Summary: {' '.join(summary.text.strip().split())}"))

        # Extract from CoreConcepts
        for concept in root.findall('.//Concept'):
            name = concept.get('name')
            desc_element = concept.find('Description') # Directly finding description within concept
            if name and desc_element is not None and desc_element.text:
                desc_text = ' '.join(desc_element.text.strip().split())
                chunks.append((f'Concept: {name}', f"EIDO Concept '{name}': {desc_text}"))

        # Extract from NIEM_Integration
        niem_integration = root.find('NIEM_Integration')
        if niem_integration is not None:
            purpose = niem_integration.find('Purpose')
            if purpose is not None and purpose.text:
                chunks.append(('NIEM Integration Purpose', f"Purpose of NIEM Integration: {' '.join(purpose.text.strip().split())}"))
            
            for child in niem_integration:
                if child.tag not in ['Purpose'] and child.text:
                    tag_name = child.tag.replace('_', ' ')
                    text_content = ' '.join(child.itertext()).strip().replace('\n', ' ').replace('  ', ' ')
                    chunks.append((f'NIEM Integration: {tag_name}', f"{tag_name} for NIEM Integration: {text_content}"))

        # Extract from DataComponents
        for component in root.findall('.//DataComponents/Component'):
            name = component.get('name')
            desc_element = component.find('Description')
            if name and desc_element is not None and desc_element.text:
                text_parts = [f"Component Name: {name}", f"Description: {' '.join(desc_element.text.strip().split())}"]
                
                fields = component.findall('Fields/Field')
                if fields:
                    text_parts.append("Key Fields:")
                    for field in fields:
                        field_name = field.get('name')
                        field_desc_element = field.find('Description')
                        if field_name and field_desc_element is not None and field_desc_element.text:
                             field_desc_text = ' '.join(field_desc_element.text.strip().split())
                             text_parts.append(f"  - {field_name}: {field_desc_text}")
                chunks.append((f'Component Context: {name}', "\n".join(text_parts)))

        logger.info(f"Created {len(chunks)} text chunks from XML context file.")
        return chunks
    except Exception as e:
        logger.error(f"Failed to parse or process XML context file {xml_path}: {e}", exc_info=True)
        return []

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
    chunk_data_list = []

    for i, (name, text) in enumerate(chunks):
        logger.debug(f"Embedding chunk {i+1}/{len(chunks)}: '{name}'")
        embedding = generate_embedding(text)
        if embedding and len(embedding) == embedding_dim:
            embeddings_list.append(embedding)
            chunk_data_list.append({"name": name, "text": text})
        elif embedding:
             logger.warning(f"Failed to generate embedding for chunk '{name}' with correct dimension ({len(embedding)} vs {embedding_dim}). Skipping.")
        else:
            logger.warning(f"Failed to generate embedding for chunk '{name}'. Skipping.")

    if not embeddings_list:
        logger.error("No embeddings were successfully generated. Index not saved.")
        return False

    logger.info(f"Generated {len(embeddings_list)} valid embeddings.")

    index_data_to_save = {
        "embedding_model": settings.embedding_model_name,
        "embedding_dim": embedding_dim,
        "chunks": chunk_data_list,
        "embeddings": embeddings_list
    }

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data_to_save, f, indent=2)
        logger.info(f"Successfully saved RAG index with {len(chunk_data_list)} total chunks to: {output_path}")
        return True
    except IOError as e:
        logger.error(f"Error saving index file to {output_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving RAG index: {e}", exc_info=True)
        return False

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting EIDO RAG Indexing ---")
    
    all_chunks = []
    
    # 1. Process OpenAPI Schema for technical details
    eido_schema = load_openapi_schema(SCHEMA_PATH)
    if eido_schema:
        schema_chunks = create_schema_chunks(eido_schema)
        if schema_chunks:
            all_chunks.extend(schema_chunks)
            logger.info(f"Added {len(schema_chunks)} chunks from OpenAPI schema.")
        else:
            logger.warning("No chunks created from OpenAPI schema.")
    else:
        logger.error(f"Failed to load EIDO schema from {SCHEMA_PATH}.")

    # 2. Process XML Context for conceptual understanding
    xml_chunks = create_xml_context_chunks(XML_CONTEXT_PATH)
    if xml_chunks:
        all_chunks.extend(xml_chunks)
        logger.info(f"Added {len(xml_chunks)} chunks from XML context.")
    else:
        logger.warning("No chunks created from XML context file.")

    # 3. Build and save the combined index
    if all_chunks:
        if build_and_save_index(all_chunks, INDEX_FILE_PATH):
             logger.info("Indexing completed successfully.")
        else:
             logger.error("Index building failed.")
    else:
        logger.error("No chunks were created from any source. Indexing aborted.")
        
    logger.info("--- RAG Indexing Finished ---")