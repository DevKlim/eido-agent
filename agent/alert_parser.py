# agent/alert_parser.py
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Any

# Import the LLM interface function we will create/use
try:
    from agent.llm_interface import extract_eido_from_alert_text
except ImportError as e:
    print(f"CRITICAL ERROR in alert_parser.py: Failed to import dependencies - {e}")
    raise SystemExit(f"Alert Parser import failed: {e}") from e

logger = logging.getLogger(__name__)

def _generate_eido_compatible_dict(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the structured data extracted by the LLM and attempts to map it
    into a dictionary resembling the EIDO structure expected by agent_core's extractor.

    Focuses on populating fields used by _extract_core_data_from_dict:
    - eidoMessageIdentifier
    - incidentComponent[0].lastUpdateTimeStamp
    - incidentComponent[0].incidentTypeCommonRegistryText
    - incidentComponent[0].incidentTrackingIdentifier (optional)
    - incidentComponent[0].locationReference ($ref:loc-1)
    - notesComponent[0].noteText
    - notesComponent[0].noteDateTimeStamp
    - locationComponent[0].$id ('loc-1')
    - locationComponent[0].locationByValue (address string or coords)
    - agencyComponent[0].$id ('agency-1')
    - agencyComponent[0].agencyName
    - incidentComponent[0].incidentOriginatingAgencyIdentifier ('agency-1') or sendingSystemIdentifier
    """
    eido_dict = {}
    report_uuid = str(uuid.uuid4())
    message_id = f"llm_parsed_{report_uuid[:8]}"
    loc_ref_id = f"loc-{report_uuid[:8]}"
    agency_ref_id = f"agency-{report_uuid[:8]}"
    note_ref_id = f"note-{report_uuid[:8]}" # Although notes aren't referenced by ID in core extraction

    # --- Top Level ---
    eido_dict['eidoMessageIdentifier'] = message_id
    eido_dict['$id'] = message_id # Add $id as fallback
    # Use current time if LLM didn't extract one reliably
    timestamp_str = extracted_data.get('timestamp_iso', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    eido_dict['lastUpdateTimeStamp'] = timestamp_str # Also put at top level as fallback

    # --- Incident Component ---
    incident_comp = {}
    incident_comp['componentIdentifier'] = f"inc-{report_uuid[:8]}"
    incident_comp['lastUpdateTimeStamp'] = timestamp_str
    incident_comp['incidentTypeCommonRegistryText'] = extracted_data.get('incident_type', 'Unknown - Parsed Alert')
    incident_comp['incidentTrackingIdentifier'] = extracted_data.get('external_id') # Optional
    incident_comp['locationReference'] = f"$ref:{loc_ref_id}" # Reference the location component

    # Agency reference - try to link, otherwise use sendingSystemIdentifier
    source_agency = extracted_data.get('source_agency')
    if source_agency:
        # incident_comp['incidentOriginatingAgencyIdentifier'] = agency_ref_id # Link to agency component
        # Let's try putting the reference in updatedByAgencyReference as seen in some examples
        incident_comp['updatedByAgencyReference'] = f"$ref:{agency_ref_id}"
    else:
        # If no specific agency, maybe use a generic source name?
        eido_dict['sendingSystemIdentifier'] = "LLM-Parser"

    eido_dict['incidentComponent'] = [incident_comp]

    # --- Notes Component ---
    description = extracted_data.get('description')
    if description:
        note_comp = {}
        note_comp['componentIdentifier'] = note_ref_id
        note_comp['noteDateTimeStamp'] = timestamp_str # Use report timestamp for the note
        note_comp['noteText'] = description
        # authorReference could be added if LLM extracts it
        eido_dict['notesComponent'] = [note_comp]
    else:
         # Add a placeholder note if description is missing but agent expects notesComponent
         eido_dict['notesComponent'] = [{
             'componentIdentifier': note_ref_id,
             'noteDateTimeStamp': timestamp_str,
             'noteText': "No detailed description extracted from alert text."
         }]


    # --- Location Component ---
    location_comp = {}
    location_comp['$id'] = loc_ref_id # Use $id for component identification
    location_comp['componentIdentifier'] = loc_ref_id # Also add componentIdentifier

    location_info = extracted_data.get('location_address') or extracted_data.get('location_description')
    coords = extracted_data.get('coordinates') # Expected as [lat, lon] list/tuple

    if coords and isinstance(coords, (list, tuple)) and len(coords) == 2:
        try:
            lat = float(coords[0])
            lon = float(coords[1])
            # Simple GML-like structure within locationByValue for coordinates
            location_comp['locationByValue'] = f"""<?xml version="1.0" encoding="UTF-8"?>
<location>
    <gml:Point xmlns:gml="http://www.opengis.net/gml">
        <gml:pos>{lat} {lon}</gml:pos>
    </gml:Point>
    <civicAddressText>{location_info or 'Coordinates provided'}</civicAddressText>
</location>"""
            logger.debug(f"Generated locationByValue with coordinates: ({lat}, {lon})")
        except (ValueError, TypeError):
             logger.warning(f"LLM provided invalid coordinates: {coords}. Using address/description only.")
             if location_info:
                 # Simple text structure if coords fail or are missing
                 location_comp['locationByValue'] = f"""<?xml version="1.0" encoding="UTF-8"?>
<location>
    <civicAddressText>{location_info}</civicAddressText>
</location>"""
             else:
                 location_comp['locationByValue'] = "Location information could not be extracted."
    elif location_info:
         # Simple text structure if only address/description is available
         location_comp['locationByValue'] = f"""<?xml version="1.0" encoding="UTF-8"?>
<location>
    <civicAddressText>{location_info}</civicAddressText>
</location>"""
         logger.debug(f"Generated locationByValue with address/description: {location_info}")
    else:
        # Fallback if no location info at all
        location_comp['locationByValue'] = "Location information could not be extracted."

    eido_dict['locationComponent'] = [location_comp]


    # --- Agency Component (if source agency extracted) ---
    if source_agency:
        agency_comp = {}
        agency_comp['$id'] = agency_ref_id
        agency_comp['agencyIdentifier'] = agency_ref_id # Use same ID
        agency_comp['agencyName'] = source_agency
        eido_dict['agencyComponent'] = [agency_comp]

    logger.info(f"Generated EIDO-compatible dictionary from LLM data for message {message_id}")
    logger.debug(f"Generated dict structure: {json.dumps(eido_dict, indent=2)}")
    return eido_dict


def parse_alert_to_eido_dict(alert_text: str) -> Optional[Dict[str, Any]]:
    """
    Takes raw alert text, uses an LLM to extract structured information,
    and formats it into an EIDO-like dictionary suitable for processing.

    Args:
        alert_text: The raw text of the emergency alert or report.

    Returns:
        A dictionary structured similarly to an EIDO message, or None if parsing fails.
    """
    if not alert_text or not isinstance(alert_text, str):
        logger.error("Invalid input: alert_text must be a non-empty string.")
        return None

    logger.info("Attempting to parse alert text using LLM...")
    logger.debug(f"Alert text received:\n--- START ALERT ---\n{alert_text}\n--- END ALERT ---")

    # Call the LLM interface function (to be implemented in llm_interface.py)
    # This function is expected to return a JSON *string* if successful
    extracted_json_str = extract_eido_from_alert_text(alert_text)

    if not extracted_json_str:
        logger.error("LLM did not return any structured data from the alert text.")
        return None

    # Parse the JSON string returned by the LLM
    try:
        extracted_data = json.loads(extracted_json_str)
        if not isinstance(extracted_data, dict):
             logger.error(f"LLM returned valid JSON, but it's not a dictionary (type: {type(extracted_data)}). Cannot process.")
             return None
        logger.info("Successfully parsed structured data from LLM response.")
        logger.debug(f"LLM Extracted Data: {extracted_data}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from LLM: {e}")
        logger.warning(f"LLM Raw Response (potential non-JSON):\n{extracted_json_str}")
        # TODO: Optionally, add a fallback mechanism here?
        # Could try regex or simpler parsing if JSON fails, but adds complexity.
        # For now, we require the LLM to return valid JSON.
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing LLM response: {e}", exc_info=True)
        return None

    # Convert the extracted data into the EIDO-like dictionary structure
    try:
        eido_compatible_dict = _generate_eido_compatible_dict(extracted_data)
        return eido_compatible_dict
    except Exception as e:
        logger.error(f"Failed to generate EIDO-compatible dictionary from extracted data: {e}", exc_info=True)
        return None

# Example Usage (for testing)
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