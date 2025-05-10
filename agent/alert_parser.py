import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Any

try:
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

    eido_dict['eidoMessageIdentifier'] = message_id
    eido_dict['$id'] = message_id # Add $id for consistency with some EIDO samples
    timestamp_str = extracted_data.get('timestamp_iso', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    eido_dict['lastUpdateTimeStamp'] = timestamp_str

    # --- Incident Component ---
    incident_comp = {}
    incident_comp['componentIdentifier'] = f"inc-{report_uuid[:8]}"
    incident_comp['lastUpdateTimeStamp'] = timestamp_str
    incident_comp['incidentTypeCommonRegistryText'] = extracted_data.get('incident_type', 'Unknown - Parsed Alert')
    incident_comp['incidentTrackingIdentifier'] = extracted_data.get('external_id') # e.g., CAD ID
    incident_comp['locationReference'] = f"$ref:{loc_ref_id}"
    
    source_agency_name = extracted_data.get('source_agency')
    if source_agency_name:
        incident_comp['updatedByAgencyReference'] = f"$ref:{agency_ref_id}" # Point to an agency component
        # Also store the sending system ID as the agency if it's the only source info
        eido_dict['sendingSystemIdentifier'] = source_agency_name # Or a URN like "urn:agency:llm-parsed-source"
    else: # If no source agency, use a generic sender
        eido_dict['sendingSystemIdentifier'] = "LLM-Parser-Agent"
    eido_dict['incidentComponent'] = [incident_comp]


    # --- Notes Component ---
    description = extracted_data.get('description')
    note_comp = {'componentIdentifier': note_ref_id, 'noteDateTimeStamp': timestamp_str}
    if description:
        note_comp['noteText'] = description
    else:
        note_comp['noteText'] = "No detailed description extracted by LLM."
    eido_dict['notesComponent'] = [note_comp]

    # --- Location Component ---
    location_comp = {}
    location_comp['$id'] = loc_ref_id
    location_comp['componentIdentifier'] = loc_ref_id
    
    location_address = extracted_data.get('location_address')
    location_description = extracted_data.get('location_description') # Broader location text
    coords = extracted_data.get('coordinates') # Expect [lat, lon]
    zip_code = extracted_data.get('zip_code')

    full_location_text = location_address or location_description or ""
    if zip_code and zip_code not in full_location_text: # Avoid duplicating ZIP if already in address
        full_location_text += f" ZIP: {zip_code}"
    full_location_text = full_location_text.strip()

    # Construct locationByValue XML
    # We'll create a simple XML structure that agent_core can parse
    # It includes gml:Point for coordinates and civicAddressText for the address string
    location_xml_parts = []
    if coords and isinstance(coords, (list, tuple)) and len(coords) == 2:
        try:
            lat, lon = float(coords[0]), float(coords[1])
            location_xml_parts.append(f'<gml:Point xmlns:gml="http://www.opengis.net/gml"><gml:pos>{lat} {lon}</gml:pos></gml:Point>')
            logger.debug(f"Generated location with coordinates: ({lat}, {lon}) for message {message_id}")
        except (ValueError, TypeError):
             logger.warning(f"Invalid coordinates from LLM: {coords} for message {message_id}. Using text only.")
             coords = None # Clear invalid coords
    
    # Add civic address text, including zip code if present and not already in full_location_text
    civic_address_text_content = full_location_text or "Location information unavailable"
    location_xml_parts.append(f"<civicAddressText>{civic_address_text_content}</civicAddressText>")
    
    # If we have structured ZIP, ensure it's captured, even if not in civicAddressText
    # The agent_core's XML parser will also look for ca:PC
    if zip_code:
        location_xml_parts.append(f'<ca:civicAddress xmlns:ca="urn:ietf:params:xml:ns:pidf:geopriv10:civicAddr"><ca:PC>{zip_code}</ca:PC></ca:civicAddress>')


    location_comp['locationByValue'] = f"""<?xml version="1.0" encoding="UTF-8"?><location>{"".join(location_xml_parts)}</location>"""
    eido_dict['locationComponent'] = [location_comp]

    # --- Agency Component (if source_agency was provided) ---
    if source_agency_name:
        agency_comp = {
            '$id': agency_ref_id,
            'agencyIdentifier': agency_ref_id, # Could be more formal if agency IDs are known/generated
            'agencyName': source_agency_name
        }
        eido_dict['agencyComponent'] = [agency_comp]

    logger.info(f"Generated EIDO-like dictionary from LLM data for message {message_id}")
    # logger.debug(f"Generated dict for {message_id}: {json.dumps(eido_dict, indent=2)}")
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
    # logger.debug(f"Alert text received for LLM parsing:\n--- START ALERT ---\n{alert_text}\n--- END ALERT ---")

    extracted_json_str = extract_eido_from_alert_text(alert_text) # This function is in llm_interface.py

    if not extracted_json_str:
        logger.error("LLM did not return structured data from the alert text.")
        return None

    try:
        # The llm_interface.extract_eido_from_alert_text should already clean and validate JSON string
        extracted_data = json.loads(extracted_json_str)
        if not isinstance(extracted_data, dict):
             logger.error(f"LLM returned JSON, but not a dictionary (type: {type(extracted_data)}). Data: {extracted_data}")
             return None
        logger.info("Successfully parsed structured data from LLM response for alert parsing.")
        # logger.debug(f"LLM Extracted Data (for EIDO generation): {extracted_data}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from LLM: {e}")
        logger.warning(f"LLM Raw Response (potential non-JSON for alert parsing):\n{extracted_json_str}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing LLM response for alert parsing: {e}", exc_info=True)
        return None

    try:
        eido_compatible_dict = _generate_eido_compatible_dict(extracted_data)
        return eido_compatible_dict
    except Exception as e:
        logger.error(f"Failed to generate EIDO-compatible dictionary from extracted data: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # This requires llm_interface.extract_eido_from_alert_text to be functional
    # and configured with a working LLM.
    # For local testing without actual LLM, mock `extract_eido_from_alert_text`
    
    # --- Mocking for local test ---
    def mock_extract_eido_from_alert_text(text: str) -> Optional[str]:
        print(f"[MOCK] LLM Interface 'extract_eido_from_alert_text' called with: '{text[:50]}...'")
        # Simulate LLM returning a JSON string with all expected fields
        return json.dumps({
            "incident_type": "Vehicle Accident",
            "timestamp_iso": "2024-05-21T15:30:00-07:00", # Example with offset
            "location_address": "Intersection of Main St and Elm Ave, Springfield",
            "coordinates": [34.0522, -118.2437],
            "zip_code": "98765",
            "description": "Two car collision, minor injuries reported. Blocking eastbound lane. Patient complains of neck pain.",
            "source_agency": "Springfield Police Dept Traffic Unit",
            "external_id": "CAD-2024-98765"
        })
    # Replace the actual function with the mock for this test run
    extract_eido_from_alert_text = mock_extract_eido_from_alert_text
    # --- End Mocking ---

    sample_alert = """
    ALERT from SPD: Vehicle collision reported at Main St / Elm Ave, Springfield around 3:30 PM PST on May 21, 2024.
    Incident # CAD-2024-98765. Two vehicles involved, reports of minor injuries, one patient with neck pain.
    Eastbound lane is blocked. Springfield Police Dept Traffic Unit 7 responding. Lat: 34.0522, Lon: -118.2437. ZIP: 98765.
    """
    result_dict = parse_alert_to_eido_dict(sample_alert)

    if result_dict:
        print("\n--- Generated EIDO-like Dictionary (from mock LLM) ---")
        print(json.dumps(result_dict, indent=2))
        print("------------------------------------")
        
        # Test the XML part specifically
        if result_dict.get('locationComponent') and isinstance(result_dict['locationComponent'], list):
            loc_val = result_dict['locationComponent'][0].get('locationByValue')
            if loc_val:
                print("\n--- LocationByValue XML ---")
                print(loc_val)
                print("---------------------------")
    else:
        print("\n--- Failed to generate dictionary ---")