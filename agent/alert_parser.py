import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any

try:
    from agent.llm_interface import fill_eido_template, choose_eido_template
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


def parse_alert_to_eido_dict(alert_text: str) -> Optional[Dict[str, Any]]:
    """
    Takes raw alert text, uses an LLM to select the best EIDO template,
    and then uses another LLM call to fill that template with the alert's data.
    """
    if not alert_text or not isinstance(alert_text, str):
        logger.error("Invalid input: alert_text must be a non-empty string.")
        return None

    # Ensure templates are loaded
    _load_templates()
    if not _template_cache:
        logger.error(
            "No EIDO templates are available. Cannot process alert text.")
        return None

    logger.info(
        "Attempting to parse single event alert text using new 'choose-then-fill' LLM workflow...")

    # 1. Choose the best template
    template_summaries_str = _get_template_summaries_str()
    chosen_template_name = choose_eido_template(
        alert_text, template_summaries_str)

    if not chosen_template_name or chosen_template_name not in _template_cache:
        logger.warning(
            f"LLM did not choose a valid template for the alert. Aborting processing for this event. Chosen: '{chosen_template_name}'")
        return None

    template_content = _template_cache[chosen_template_name]

    # 2. Fill the chosen template
    logger.info(
        f"Attempting to fill template '{chosen_template_name}' with alert text.")
    # The alert text itself serves as the scenario description
    filled_eido_json_str = fill_eido_template(template_content, alert_text)

    if not filled_eido_json_str:
        logger.error(
            f"LLM failed to fill the chosen EIDO template '{chosen_template_name}'.")
        return None

    try:
        eido_dict = json.loads(filled_eido_json_str)
        if not isinstance(eido_dict, dict):
            logger.error(
                f"LLM returned JSON, but not a dictionary (type: {type(eido_dict)}). Data: {eido_dict}")
            return None
        logger.info(
            f"Successfully generated EIDO dictionary from alert using template '{chosen_template_name}'.")

        # The following diagnostic prints are commented out as they are for development/debugging
        # and should not be active in production code.
        # message_id = eido_dict.get('eidoMessageIdentifier', 'N/A')
        # logger.debug(
        #     f"DIAGNOSTIC: EIDO Dict from Template (message_id: {message_id})")
        # print(json.dumps(eido_dict, indent=2))
        # print("="*80 + "\n")

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Mock the LLM calls for a standalone test
    def mock_choose_eido_template(text: str, summaries: str) -> Optional[str]:
        print(
            f"[MOCK] LLM 'choose_eido_template' called with: '{text[:50]}...'")
        if "fire" in text.lower():
            return "ucsd_vegetation_fire_template.json"
        if "collision" in text.lower():
            return "traffic_collision.json"
        return None

    def mock_fill_eido_template(template: str, scenario: str) -> Optional[str]:
        print(
            f"[MOCK] LLM 'fill_eido_template' called for scenario: '{scenario[:50]}...'")
        # Just return a valid JSON string for testing the flow
        return json.dumps({
            "$id": "urn:emergency:uid:incidentid:mock-filled:bcf.state.pa.us",
            "lastUpdateTimeStamp": datetime.now(timezone.utc).isoformat(),
            "incidentComponent": [{
                "incidentTypeCommonRegistryText": "Mock Filled Incident",
                "incidentSummaryText": f"This is a mocked response for the scenario: {scenario}"
            }]
        })

    # Replace the actual LLM calls with mocks for the test
    original_choose_eido_template = choose_eido_template
    original_fill_eido_template = fill_eido_template
    choose_eido_template = mock_choose_eido_template
    fill_eido_template = mock_fill_eido_template

    # Create dummy template files for the mock _load_templates to find
    mock_template_dir = os.path.join(
        os.path.dirname(__file__), '..', 'eido_templates')
    os.makedirs(mock_template_dir, exist_ok=True)

    # Dummy content for the mock templates
    with open(os.path.join(mock_template_dir, "ucsd_vegetation_fire_template.json"), "w") as f:
        f.write(json.dumps({"description": "Template for a vegetation fire incident.",
                "incidentComponent": [{"incidentTypeCommonRegistryText": "Vegetation Fire"}]}))
    with open(os.path.join(mock_template_dir, "traffic_collision.json"), "w") as f:
        f.write(json.dumps({"description": "Template for a traffic collision incident.",
                "incidentComponent": [{"incidentTypeCommonRegistryText": "Traffic Collision"}]}))

    # Clear cache to force _load_templates to run and find the mock files
    _template_cache.clear()
    _template_summaries.clear()

    sample_alert = "ALERT from UCPD: Report of a small brush fire near the canyon."
    result_dict = parse_alert_to_eido_dict(sample_alert)

    if result_dict:
        print("\n--- Successfully parsed alert text into EIDO dict ---")
        print(json.dumps(result_dict, indent=2))
    else:
        print("\n--- Failed to parse alert text ---")

    # Clean up dummy template files and directory
    os.remove(os.path.join(mock_template_dir,
              "ucsd_vegetation_fire_template.json"))
    os.remove(os.path.join(mock_template_dir, "traffic_collision.json"))
    try:
        os.rmdir(mock_template_dir)
    except OSError:
        pass  # Directory not empty, or other error

    # Restore original functions
    choose_eido_template = original_choose_eido_template
    fill_eido_template = original_fill_eido_template
