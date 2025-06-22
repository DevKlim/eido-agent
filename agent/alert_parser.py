import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any

try:
    from agent.llm_interface import fill_eido_template, choose_eido_template, PROMPTS
except ImportError as e:
    print(f"CRITICAL ERROR in alert_parser.py: {e}")
    raise SystemExit(f"Alert Parser import failed: {e}") from e

logger = logging.getLogger(__name__)


# --- Template Loading Helpers ---
EIDO_TEMPLATE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'eido_templates'))
_template_cache: Dict[str, str] = {}
_template_summaries: Dict[str, str] = {}


def _load_templates():
    """Loads all EIDO templates from the specified directory into a cache."""
    if _template_cache:  # Already loaded
        return
    if not os.path.isdir(EIDO_TEMPLATE_DIR):
        logger.error(f"EIDO template directory not found: {EIDO_TEMPLATE_DIR}")
        return

    for filename in os.listdir(EIDO_TEMPLATE_DIR):
        if filename.endswith(".json"):
            try:
                filepath = os.path.join(EIDO_TEMPLATE_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    _template_cache[filename] = content
                    # Create a simple summary for the LLM
                    try:
                        template_json = json.loads(content)
                        # Look for a descriptive field, e.g., in a top-level "description" or from the incident type
                        desc = template_json.get("description", "")
                        if not desc:
                            inc_comp = template_json.get(
                                "incidentComponent", [{}])[0]
                            inc_type = inc_comp.get(
                                "incidentTypeCommonRegistryText", f"Generic incident ({filename})")
                            desc = f"Template for a '{inc_type}' incident."
                        _template_summaries[filename] = desc
                    except json.JSONDecodeError:
                        _template_summaries[
                            filename] = f"A general EIDO template named {filename}."

            except Exception as e:
                logger.error(
                    f"Failed to load or parse EIDO template '{filename}': {e}")

    logger.info(
        f"Loaded {len(_template_cache)} EIDO templates from {EIDO_TEMPLATE_DIR}")


def _get_template_summaries_str() -> str:
    """Returns a formatted string of template names and their summaries."""
    if not _template_summaries:
        _load_templates()

    return "\n".join([f"- {filename}: {_template_summaries[filename]}" for filename in sorted(_template_summaries.keys())])


async def parse_alert_to_eido_dict(alert_text: str) -> Optional[Dict[str, Any]]:
    """
    Takes raw alert text, uses an LLM to select the best EIDO template, and then uses
    another LLM call to fill that template. If no template is found or chosen, it
    resorts to filling a generic, hardcoded template as a fallback.
    """
    if not alert_text or not isinstance(alert_text, str):
        logger.error("Invalid input: alert_text must be a non-empty string.")
        return None

    _load_templates()

    template_content = None
    chosen_template_name_for_log = "N/A"

    # 1. Try to choose a template from the filesystem
    if _template_cache:
        template_summaries_str = _get_template_summaries_str()
        chosen_template_name = await choose_eido_template(
            alert_text, template_summaries_str)
        if chosen_template_name and chosen_template_name in _template_cache:
            template_content = _template_cache[chosen_template_name]
            chosen_template_name_for_log = chosen_template_name
    
    # 2. If no template was chosen or available, resort to the generic fallback
    if template_content is None:
        if not _template_cache:
            logger.warning(
                "No EIDO templates found in directory. Resorting to generic in-memory template from prompt library.")
        else:
            logger.warning(
                f"LLM did not choose a valid template. Resorting to generic in-memory template from prompt library."
            )
        
        template_content = PROMPTS.get("GENERIC_EIDO_TEMPLATE")
        if not template_content:
            logger.error("FATAL: Generic EIDO template is missing from prompt_library.json. Cannot process alert.")
            return None # Critical failure
        
        chosen_template_name_for_log = "generic_fallback"
    
    # 3. Fill the chosen or fallback template
    logger.info(
        f"Attempting to fill template '{chosen_template_name_for_log}' with alert text.")
    filled_eido_json_str = await fill_eido_template(template_content, alert_text)

    if not filled_eido_json_str:
        logger.error(
            f"LLM failed to fill the chosen template '{chosen_template_name_for_log}'.")
        return None

    try:
        eido_dict = json.loads(filled_eido_json_str)
        if not isinstance(eido_dict, dict):
            logger.error(
                f"LLM returned JSON, but not a dictionary (type: {type(eido_dict)}). Data: {eido_dict}")
            return None
        logger.info(
            f"Successfully generated EIDO dictionary from alert using template '{chosen_template_name_for_log}'.")
        
        return eido_dict
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse JSON response from LLM after filling template: {e}")
        logger.warning(
            f"LLM Raw Response (potential non-JSON for template fill):\n{filled_eido_json_str}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error processing LLM template fill response: {e}", exc_info=True)
        return None