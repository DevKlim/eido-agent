# agent/alert_parser.py
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Any

try:
    # Use the updated llm_interface function
    from agent.llm_interface import extract_eido_from_alert_text
except ImportError as e:
    print(f"CRITICAL ERROR in alert_parser.py: {e}"); raise SystemExit(f"Alert Parser import failed: {e}") from e

logger = logging.getLogger(__name__)

def _generate_eido_compatible_dict(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """ Maps extracted LLM data to an EIDO-like dictionary. """
    eido_dict = {}
    report_uuid = str(uuid.uuid4())
    message_id = f"llm_parsed_{report_uuid[:8]}"
    loc_ref_id = f"loc-{report_uuid[:8]}"
    agency_ref_id = f"agency-{report_uuid[:8]}"
    note_ref_id = f"note-{report_uuid[:8]}"

    # --- Top Level ---
    eido_dict['eidoMessageIdentifier'] = message_id
    eido_dict['$id'] = message_id
    timestamp_str = extracted_data.get('timestamp_iso', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    eido_dict['lastUpdateTimeStamp'] = timestamp_str

    # --- Incident Component ---
    incident_comp = {}
    incident_comp['componentIdentifier'] = f"inc-{report_uuid[:8]}"
    incident_comp['lastUpdateTimeStamp'] = timestamp_str
    incident_comp['incidentTypeCommonRegistryText'] = extracted_data.get('incident_type', 'Unknown - Parsed Alert')
    incident_comp['incidentTrackingIdentifier'] = extracted_data.get('external_id')
    incident_comp['locationReference'] = f"$ref:{loc_ref_id}"
    source_agency = extracted_data.get('source_agency')
    if source_agency: incident_comp['updatedByAgencyReference'] = f"$ref:{agency_ref_id}"
    else: eido_dict['sendingSystemIdentifier'] = "LLM-Parser"
    eido_dict['incidentComponent'] = [incident_comp]

    # --- Notes Component ---
    description = extracted_data.get('description')
    note_comp = {'componentIdentifier': note_ref_id, 'noteDateTimeStamp': timestamp_str}
    if description: note_comp['noteText'] = description
    else: note_comp['noteText'] = "No detailed description extracted."
    eido_dict['notesComponent'] = [note_comp]

    # --- Location Component ---
    location_comp = {}
    location_comp['$id'] = loc_ref_id
    location_comp['componentIdentifier'] = loc_ref_id
    location_info = extracted_data.get('location_address') or extracted_data.get('location_description')
    coords = extracted_data.get('coordinates')
    # --- Include zip code in location info if available ---
    zip_code = extracted_data.get('zip_code')
    full_location_text = location_info or ""
    if zip_code:
        full_location_text += f" (ZIP: {zip_code})"
    full_location_text = full_location_text.strip()
    # --- End zip code ---

    location_xml_content = ""
    if coords and isinstance(coords, (list, tuple)) and len(coords) == 2:
        try:
            lat, lon = float(coords[0]), float(coords[1])
            location_xml_content += f'<gml:Point xmlns:gml="http://www.opengis.net/gml"><gml:pos>{lat} {lon}</gml:pos></gml:Point>'
            logger.debug(f"Generated location with coordinates: ({lat}, {lon})")
        except (ValueError, TypeError):
             logger.warning(f"Invalid coordinates from LLM: {coords}. Using text only.")
             coords = None # Clear invalid coords

    # Add civic address text, including zip code if present
    location_xml_content += f"<civicAddressText>{full_location_text or 'Location info unavailable'}</civicAddressText>"

    # Basic XML structure for locationByValue
    location_comp['locationByValue'] = f"""<?xml version="1.0" encoding="UTF-8"?><location>{location_xml_content}</location>"""
    eido_dict['locationComponent'] = [location_comp]

    # --- Agency Component ---
    if source_agency:
        agency_comp = {'$id': agency_ref_id, 'agencyIdentifier': agency_ref_id, 'agencyName': source_agency}
        eido_dict['agencyComponent'] = [agency_comp]

    logger.info(f"Generated EIDO-like dictionary from LLM data for message {message_id}")
    # logger.debug(f"Generated dict: {json.dumps(eido_dict, indent=2)}") # Can be verbose
    return eido_dict


def parse_alert_to_eido_dict(alert_text: str) -> Optional[Dict[str, Any]]:
    """
    Takes raw alert text for a SINGLE event, uses LLM to extract structured info,
    and formats it into an EIDO-like dictionary.
    """
    if not alert_text or not isinstance(alert_text, str):
        logger.error("Invalid input: alert_text must be a non-empty string.")
        return None

    logger.info("Attempting to parse single event alert text using LLM...")
    logger.debug(f"Alert text received:\n--- START ALERT ---\n{alert_text}\n--- END ALERT ---")

    # Call the LLM to extract structured JSON *string*
    extracted_json_str = extract_eido_from_alert_text(alert_text)

    if not extracted_json_str:
        logger.error("LLM did not return structured data from the alert text.")
        return None

    try:
        extracted_data = json.loads(extracted_json_str)
        if not isinstance(extracted_data, dict):
             logger.error(f"LLM returned JSON, but not a dictionary (type: {type(extracted_data)}).")
             return None
        logger.info("Successfully parsed structured data from LLM response.")
        logger.debug(f"LLM Extracted Data: {extracted_data}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from LLM: {e}")
        logger.warning(f"LLM Raw Response (potential non-JSON):\n{extracted_json_str}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing LLM response: {e}", exc_info=True)
        return None

    try:
        eido_compatible_dict = _generate_eido_compatible_dict(extracted_data)
        return eido_compatible_dict
    except Exception as e:
        logger.error(f"Failed to generate EIDO-compatible dictionary: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # This requires llm_interface.extract_eido_from_alert_text to be functional
    # and configured with a working LLM.
    logging.basicConfig(level=logging.DEBUG)
    # Mock llm_interface for local testing without actual LLM call
    def mock_extract(text):
        print("[MOCK] LLM Interface called.")
        # Simulate LLM returning structured JSON
        return json.dumps({
            "incident_type": "Vehicle Accident",
            "timestamp_iso": "2024-05-21T15:30:00Z",
            "location_address": "Intersection of Main St and Elm Ave",
            "coordinates": [34.0522, -118.2437],
            "description": "Two car collision, minor injuries reported. Blocking eastbound lane.",
            "source_agency": "Traffic Unit 7",
            "external_id": "CAD-2024-98765"
        })
    extract_eido_from_alert_text = mock_extract # Override with mock

    sample_alert = """
    ALERT: Vehicle collision reported at Main St / Elm Ave around 3:30 PM PST on May 21, 2024.
    Incident # CAD-2024-98765. Two vehicles involved, reports of minor injuries.
    Eastbound lane is blocked. Traffic Unit 7 responding. Lat: 34.0522, Lon: -118.2437.
    """
    result_dict = parse_alert_to_eido_dict(sample_alert)

    if result_dict:
        print("\n--- Generated EIDO-like Dictionary ---")
        print(json.dumps(result_dict, indent=2))
        print("------------------------------------")
    else:
        print("\n--- Failed to generate dictionary ---")