import yaml
import json
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)
# Configure logger if not already configured by a higher-level basicConfig
if not logger.hasHandlers():
    # Basic config for direct script run or if root logger not configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# Define path relative to this script file
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(UTILS_DIR, '..'))
DEFAULT_SCHEMA_PATH = '/app/schema/openapi.yaml'

def load_openapi_schema(file_path: str = DEFAULT_SCHEMA_PATH) -> Optional[Dict[str, Any]]:
    """Loads the OpenAPI YAML schema."""
    logger.info(f"Attempting to load OpenAPI schema from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Schema file not found at the specified path: {file_path}")
        # Try to provide more context if path seems wrong
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Project root deduced as: {PROJECT_ROOT}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
        if schema and isinstance(schema, dict):
            logger.info(f"Successfully loaded and parsed OpenAPI schema from: {file_path}")
            return schema
        else:
             logger.error(f"Failed to parse YAML or schema is not a dictionary in {file_path}.")
             return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML schema file {file_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading schema from {file_path}: {e}", exc_info=True)
        return None

def get_component_definition(schema: Dict[str, Any], component_name: str) -> Optional[Dict[str, Any]]:
    """Retrieves the raw schema definition for a component by name."""
    if not schema or 'components' not in schema or 'schemas' not in schema['components']:
        logger.error("Invalid schema structure: missing 'components/schemas'.")
        return None
    
    component_def = schema['components']['schemas'].get(component_name)
    if component_def is None:
         logger.debug(f"Component '{component_name}' not found directly in schema['components']['schemas'].")
    return component_def

def format_component_details_for_llm(schema: Dict[str, Any], component_name: str) -> Optional[str]:
    """Formats component details into a string suitable for LLM RAG context."""
    component_schema = get_component_definition(schema, component_name)
    if not component_schema:
        # logger.warning(f"Component definition for '{component_name}' not found in schema.")
        return None # Return None if component doesn't exist

    details_lines = []
    details_lines.append(f"Component Name: {component_name}")

    # --- Description ---
    if component_schema.get('description'):
        # Clean up description formatting if needed (e.g., remove excessive newlines)
        desc = ' '.join(component_schema['description'].split())
        details_lines.append(f"Description: {desc}")

    # --- Type (if explicitly defined, e.g., 'object', 'string') ---
    comp_type = component_schema.get('type')
    if comp_type:
         details_lines.append(f"Type: {comp_type}")
    
    properties = {}
    required_fields = []
    # --- allOf ---
    if 'allOf' in component_schema:
        for part in component_schema['allOf']:
            if '$ref' in part:
                 ref_name = part['$ref'].split('/')[-1]
                 details_lines.append(f"Inherits from: {ref_name}")
            if 'properties' in part:
                 properties.update(part['properties'])
            if 'required' in part:
                 required_fields.extend(part['required'])

    # --- Direct properties/required ---
    if 'properties' in component_schema:
         properties.update(component_schema['properties'])
    if 'required' in component_schema:
        required_fields.extend(component_schema['required'])


    # --- Required Fields ---
    if required_fields:
        required_unique = sorted(list(set(required_fields))) # Sort for consistency
        details_lines.append(f"Required Fields: [{', '.join(required_unique)}]")

    # --- Properties ---
    if properties:
        details_lines.append("Properties:")
        # Sort properties for consistent output
        for prop_name in sorted(properties.keys()):
            prop_schema = properties[prop_name]
            prop_line_parts = [f"  - {prop_name}:"]

            prop_type = prop_schema.get('type')
            prop_format = prop_schema.get('format')
            prop_ref = prop_schema.get('$ref')
            prop_items = prop_schema.get('items') # For arrays
            prop_desc = prop_schema.get('description', '').strip()
            prop_desc_clean = ' '.join(prop_desc.split()) # Clean description

            if prop_ref:
                ref_name = prop_ref.split('/')[-1]
                prop_line_parts.append(f"(Reference to component '{ref_name}')")
            elif prop_type == 'array' and prop_items:
                 item_type = prop_items.get('type')
                 item_ref = prop_items.get('$ref')
                 if item_ref:
                     item_ref_name = item_ref.split('/')[-1]
                     prop_line_parts.append(f"type=array (items are references to '{item_ref_name}')")
                 elif item_type:
                      prop_line_parts.append(f"type=array (items type: {item_type})")
                 else:
                      prop_line_parts.append("type=array (items type: any)")
            elif prop_type:
                 prop_line_parts.append(f"type={prop_type}")
                 if prop_format: prop_line_parts.append(f"(format: {prop_format})")
            else: # No type or ref
                 prop_line_parts.append("(type: any)")

            if prop_desc_clean:
                 prop_line_parts.append(f" # {prop_desc_clean}")

            details_lines.append(" ".join(prop_line_parts))

    # --- Enum Values (if defined) ---
    if component_schema.get('enum'):
        enum_values = [str(val) for val in component_schema['enum']] # Ensure string representation
        details_lines.append(f"Allowed Values (Enum): [{', '.join(enum_values)}]")

    return "\n".join(details_lines).strip()


# --- Example Usage ---
if __name__ == "__main__":
    logger.info("Running Schema Parser directly for testing...")
    eido_schema = load_openapi_schema()

    if eido_schema:
        num_components = len(eido_schema.get('components', {}).get('schemas', {}))
        print(f"\nSchema loaded successfully. Found {num_components} components.")

        components_to_test = [
            "EmergencyIncidentDataObjectType",
            "IncidentInformationType",
            "LocationInformationType",
            "NotesType",
            "AgencyType",
            "ReferenceType",
            "NonExistentComponent" # Test case for missing component
        ]

        print("\n--- Formatting details for selected components ---")
        for comp_name in components_to_test:
            print(f"\n--- Component: {comp_name} ---")
            details_str = format_component_details_for_llm(eido_schema, comp_name)
            if details_str:
                print(details_str)
            else:
                print(f"-> Could not retrieve or format details for '{comp_name}'.")

    else:
        print("\nFailed to load EIDO schema. Cannot perform tests.")