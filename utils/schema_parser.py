# utils/schema_parser.py
import yaml
import json
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# Assumes script is run from project root, or adjust path
SCHEMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EIDO-JSON', 'Schema', 'openapi.yaml'))

def load_openapi_schema(file_path: str = SCHEMA_PATH) -> Optional[Dict[str, Any]]:
    """Loads the OpenAPI YAML schema."""
    if not os.path.exists(file_path):
        logger.error(f"Schema file not found at: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
        logger.info(f"Successfully loaded OpenAPI schema from: {file_path}")
        return schema
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML schema: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading schema: {e}", exc_info=True)
        return None

def get_component_definition(schema: Dict[str, Any], component_name: str) -> Optional[Dict[str, Any]]:
    """Retrieves the raw schema definition for a component."""
    if not schema or 'components' not in schema or 'schemas' not in schema['components']:
        logger.error("Invalid schema structure: missing components/schemas.")
        return None
    return schema['components']['schemas'].get(component_name)

def format_component_details_for_llm(schema: Dict[str, Any], component_name: str) -> Optional[str]:
    """Formats component details into a string suitable for LLM context."""
    component_schema = get_component_definition(schema, component_name)
    if not component_schema:
        logger.warning(f"Component '{component_name}' not found in schema.")
        return None

    details = f"Component: {component_name}\n"
    if component_schema.get('description'):
        details += f"Description: {component_schema['description']}\n"
    if component_schema.get('required'):
        details += f"Required Fields: {', '.join(component_schema['required'])}\n"

    if 'properties' in component_schema:
        details += "Properties:\n"
        for prop_name, prop_schema in component_schema['properties'].items():
            prop_type = prop_schema.get('type', 'any')
            prop_format = prop_schema.get('format')
            prop_ref = prop_schema.get('$ref')
            prop_desc = prop_schema.get('description', '').strip()

            details += f"  - {prop_name}:"
            if prop_ref:
                ref_name = prop_ref.split('/')[-1]
                details += f" (Reference to {ref_name})"
            else:
                details += f" type={prop_type}"
                if prop_format:
                    details += f" (format: {prop_format})"
            if prop_desc:
                details += f" # {prop_desc}"
            details += "\n"
    return details.strip()


# --- Example Usage ---
if __name__ == "__main__":
    eido_schema = load_openapi_schema()

    if eido_schema:
        print(f"Schema loaded. Found {len(eido_schema.get('components', {}).get('schemas', {}))} components.")

        # Use names identified from your previous output
        components_to_show = [
            "EmergencyIncidentDataObjectType", # The main object
            "IncidentInformationType",         # Likely contains incident details
            "LocationInformationType",         # Likely contains location details
            "NotesType",
            "AgencyType",
            "ReferenceType"                    # Often used for $ref structures
        ]

        for comp_name in components_to_show:
            print(f"\n--- Details for {comp_name} ---")
            details_str = format_component_details_for_llm(eido_schema, comp_name)
            if details_str:
                print(details_str)
            else:
                print(f"Could not retrieve details for {comp_name}.")

        # List all components again for verification
        # print("\n--- All Components Found ---")
        # all_components = list(eido_schema.get('components', {}).get('schemas', {}).keys())
        # for name in sorted(all_components):
        #      print(f"- {name}")