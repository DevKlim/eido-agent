# agent/agent_core.py
import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
import uuid
import xml.etree.ElementTree as ET # Import XML parser

# Use the simplified schemas
try:
    from data_models.schemas import ReportCoreData, Incident
    from services.storage import incident_store
    from services.geocoding import get_coordinates
    from services.embedding import generate_embedding
    from agent.matching import find_match_for_report
    from agent.llm_interface import summarize_incident, recommend_actions, split_raw_text_into_events, extract_eido_from_alert_text
    from agent.alert_parser import parse_alert_to_eido_dict
except ImportError as e:
    print(f"CRITICAL ERROR in agent_core.py: {e}"); raise SystemExit(f"Agent Core import failed: {e}") from e

logger = logging.getLogger(__name__)

class EidoAgent:
    """
    The core agent responsible for processing EIDO messages (as dicts) OR raw alert text,
    managing incidents, and interacting with LLMs.
    Bypasses strict EidoMessage validation to handle real-world variations.
    """

    def __init__(self):
        logger.info("EIDO Agent Core initialized.")

    def _safe_get(self, data: Optional[Dict], path: List[str], default: Any = None) -> Any:
        """Safely get a nested value from a dict using a list of keys."""
        current = data
        if not isinstance(data, dict):
            return default
        try:
            for key in path:
                if isinstance(current, list):
                    # If we encounter a list, try accessing the first element if path continues
                    if not current: return default # Empty list
                    current = current[0] # Assume we want the first item
                    if not isinstance(current, dict): return default # Item is not a dict
                # Now expect current to be a dict
                if not isinstance(current, dict): return default # Safety check
                current = current.get(key)
                if current is None:
                    return default
            return current
        except (TypeError, IndexError, KeyError):
            return default

    def _resolve_ref_string_from_dict(self, ref_input: Any) -> Optional[str]:
        """
        Helper to try and extract a string reference like '$ref:id'
        from various common input formats found in EIDO samples (dicts, lists).
        Returns the ID part (e.g., 'state.pa.us') or None.
        """
        ref_id = None
        if isinstance(ref_input, str) and ref_input.startswith("$ref:"):
            ref_id = ref_input.split(':', 1)[1]
        elif isinstance(ref_input, dict) and '$ref' in ref_input and isinstance(ref_input['$ref'], str):
            ref_id = ref_input['$ref']
        elif isinstance(ref_input, list) and ref_input:
            first_item = ref_input[0]
            if isinstance(first_item, dict) and '$ref' in first_item and isinstance(first_item['$ref'], str):
                ref_id = first_item['$ref']
                if len(ref_input) > 1:
                    logger.warning(f"Reference field contained multiple items ({len(ref_input)}), using first $ref: '{ref_id}'")
            elif isinstance(first_item, str) and first_item.startswith("$ref:"):
                ref_id = first_item.split(':', 1)[1]
                if len(ref_input) > 1:
                    logger.warning(f"Reference field contained multiple strings, using first $ref: '{ref_id}'")

        # if ref_id: logger.debug(f"Resolved reference '{ref_input}' to ID '{ref_id}'")
        # else: logger.warning(f"Could not resolve reference: {ref_input}")
        return ref_id

    def _extract_core_data_from_dict(self, eido_dict: Dict) -> Optional[ReportCoreData]:
        """
        Extracts key information directly from an EIDO message dictionary
        into the simplified ReportCoreData structure.
        Uses safe access methods (.get) to handle missing fields gracefully.
        """
        # --- >> ADDED: Initial Input Validation << ---
        if not isinstance(eido_dict, dict):
            logger.error(f"Input to _extract_core_data_from_dict is not a dictionary (Type: {type(eido_dict)}). Cannot extract.")
            return None
        # --- >> END ADDED << ---

        message_id = eido_dict.get('eidoMessageIdentifier', eido_dict.get('$id'))
        if not message_id:
            message_id = f"unknown_{str(uuid.uuid4())[:8]}"
            logger.warning(f"EIDO message dictionary missing 'eidoMessageIdentifier' and '$id'. Assigned temporary ID: {message_id}")
        logger.debug(f"Extracting core data from EIDO Message Dict: {message_id}")

        # --- Determine Primary Component ---
        incident_comp_list = eido_dict.get('incidentComponent')
        call_comp_list = eido_dict.get('callComponent')
        primary_comp = None
        source_component_type = "Unknown"

        # Use first valid dictionary found in incidentComponent list
        if isinstance(incident_comp_list, list):
            for comp in incident_comp_list:
                if isinstance(comp, dict):
                    primary_comp = comp
                    source_component_type = "IncidentComponent"
                    if len(incident_comp_list) > 1:
                         logger.warning(f"Msg {message_id}: Multiple IncidentComponents found ({len(incident_comp_list)}). Using first valid dictionary.")
                    break # Use the first valid one

        # Fallback to callComponent if no valid incidentComponent found
        if not primary_comp and isinstance(call_comp_list, list):
             for comp in call_comp_list:
                 if isinstance(comp, dict):
                     primary_comp = comp
                     source_component_type = "CallComponent"
                     logger.warning(f"Msg {message_id}: No valid IncidentComponent found. Using CallComponent for core data.")
                     if len(call_comp_list) > 1:
                          logger.warning(f"Msg {message_id}: Multiple CallComponents found ({len(call_comp_list)}). Using first valid dictionary.")
                     break # Use the first valid one

        # Check if a valid primary component was found
        if not primary_comp: return None # Added check

        # --- Extract Fields using safe .get() from the guaranteed dictionary primary_comp ---
        incident_tracking_id = primary_comp.get('incidentTrackingIdentifier')
        if not incident_tracking_id and source_component_type == "CallComponent":
             incident_tracking_id = primary_comp.get('callTrackingIdentifier') # Fallback for calls

        timestamp_str = primary_comp.get('lastUpdateTimeStamp', eido_dict.get('lastUpdateTimeStamp'))
        timestamp = ReportCoreData.ensure_timezone_aware(timestamp_str or datetime.now(timezone.utc))

        # Incident Type (handle potential list or string)
        raw_incident_type = primary_comp.get('incidentTypeCommonRegistryText')
        if not raw_incident_type and source_component_type == "CallComponent":
             raw_incident_type = primary_comp.get('callTypeCommonRegistryText', "Inferred Call")
        incident_type = None
        if isinstance(raw_incident_type, list) and raw_incident_type:
            # Ensure the first item is usable as string
            first_type = raw_incident_type[0]
            incident_type = str(first_type) if first_type is not None else "Unknown"
        elif isinstance(raw_incident_type, str):
            incident_type = raw_incident_type
        else:
            incident_type = "Unknown" # Default if not list or string

        # Location Reference ID
        loc_ref_id = self._resolve_ref_string_from_dict(primary_comp.get('locationReference'))
        agency_ref_id = self._resolve_ref_string_from_dict(primary_comp.get('incidentOriginatingAgencyIdentifier') or primary_comp.get('updatedByAgencyReference')) or eido_dict.get('sendingSystemIdentifier')

        # Source Agency Reference ID
        # Prefer originating, then updatedBy, then fallback to message-level sender
        agency_ref_input = primary_comp.get('incidentOriginatingAgencyIdentifier') # Often a direct ID string?
        if not agency_ref_input:
             agency_ref_input = primary_comp.get('updatedByAgencyReference')
        agency_ref_id = self._resolve_ref_string_from_dict(agency_ref_input)
        if not agency_ref_id: # If still no ID, try the top-level sending system
             agency_ref_id = eido_dict.get('sendingSystemIdentifier')
             if agency_ref_id: logger.debug(f"Using sendingSystemIdentifier '{agency_ref_id}' as fallback source agency ID.")


        # --- Extract Description (Combine Notes/Comments) ---
        descriptions = []
        notes_list = eido_dict.get('notesComponent', [])
        if isinstance(notes_list, list):
             # --- >> ADDED: Check item type inside loop << ---
             for note_comp in notes_list:
                 if isinstance(note_comp, dict): # Process only if it's a dictionary
                     note_text = note_comp.get('noteText')
                     if note_text: # Ensure noteText exists
                         note_ts_str = note_comp.get('noteDateTimeStamp')
                         note_ts = ReportCoreData.ensure_timezone_aware(note_ts_str or timestamp) # Use note ts or component ts
                         timestamp_str_fmt = note_ts.strftime('%Y-%m-%d %H:%M:%S Z')
                         author_ref_id = self._resolve_ref_string_from_dict(note_comp.get('authorReference'))
                         author_info = f" (AuthorRef: {author_ref_id})" if author_ref_id else ""
                         descriptions.append(f"[{timestamp_str_fmt}]{author_info}: {note_text}")
                 else:
                     logger.warning(f"Msg {message_id}: Found non-dictionary item in notesComponent: {type(note_comp)}. Skipping.")
             # --- >> END ADDED << ---

        # Add similar loop for eido_dict.get('commentsComponent', []) if needed

        full_description = "\n".join(descriptions) if descriptions else primary_comp.get('incidentSummaryText') # Fallback
        if not full_description: full_description = "No description or notes found."


        # --- Location Extraction (Refined) ---
        location_address: Optional[str] = None
        location_coords: Optional[Tuple[float, float]] = None
        zip_code: Optional[str] = None # Initialize zip_code
        primary_location_comp = None
        location_components = eido_dict.get('locationComponent', [])

        # Find the relevant location component (using ref or first valid)
        if isinstance(location_components, list):
            found_loc = False
            for loc_comp in location_components:
                if isinstance(loc_comp, dict):
                    comp_id = loc_comp.get('componentIdentifier', loc_comp.get('$id'))
                    if loc_ref_id and comp_id == loc_ref_id:
                        primary_location_comp = loc_comp; found_loc = True; break
            # If ref not found or no ref, use the first valid dict
            if not found_loc and not loc_ref_id:
                 for loc_comp in location_components:
                     if isinstance(loc_comp, dict):
                         primary_location_comp = loc_comp; break

        # Process the found location component
        if primary_location_comp:
            loc_comp_id_for_log = primary_location_comp.get('componentIdentifier', primary_location_comp.get('$id', 'N/A'))
            logger.debug(f"Processing LocationComponent: {loc_comp_id_for_log}")

            loc_val = primary_location_comp.get('locationByValue')
            loc_ref_url = primary_location_comp.get('locationReferenceUrl') # Handle this too

            # --- Try XML Parsing First (for coords, address, zip) ---
            if loc_val and isinstance(loc_val, str) and loc_val.strip().startswith('<?xml'):
                logger.debug(f"Attempting XML parse for locationByValue in {loc_comp_id_for_log}")
                try:
                    namespaces = {'ca': 'urn:ietf:params:xml:ns:pidf:geopriv10:civicAddr', 'gml': 'http://www.opengis.net/gml'}
                    # Handle potential wrapper like <location> used by alert_parser
                    root_str = loc_val.strip()
                    if root_str.startswith('<location>'): # Basic check for our wrapper
                        try:
                            # Parse the outer wrapper first
                            outer_root = ET.fromstring(root_str)
                            # Find the actual content inside (gml:Point or civicAddressText)
                            gml_point = outer_root.find('.//gml:Point', namespaces)
                            civic_text_node = outer_root.find('.//civicAddressText') # No namespace needed for this simple tag
                            ca_node = outer_root.find('.//ca:civicAddress', namespaces) # Check for full civic address too

                            if gml_point is not None:
                                pos_node = gml_point.find('.//gml:pos', namespaces)
                                if pos_node is not None and pos_node.text:
                                    coords_text = pos_node.text.strip().split()
                                    if len(coords_text) >= 2:
                                        try: location_coords = (float(coords_text[0]), float(coords_text[1])); logger.debug(f"Extracted coords from <gml:pos>: {location_coords}")
                                        except (ValueError, TypeError): logger.warning(f"Could not parse coords from <gml:pos>: '{pos_node.text}'")
                            # Extract address/zip from civicAddressText or full civicAddress
                            if ca_node is not None: # Prefer structured civic address
                                addr_parts = [ca_node.findtext(f'ca:{tag}', None, namespaces) for tag in ['HNO', 'HNS', 'PRD', 'RD', 'STS', 'POD', 'A6', 'A3', 'A1', 'PC']]
                                location_address = " ".join(filter(None, addr_parts)).strip()
                                zip_code = ca_node.findtext('ca:PC', None, namespaces)
                                logger.debug(f"Extracted address from <ca:civicAddress>: {location_address}, ZIP: {zip_code}")
                            elif civic_text_node is not None and civic_text_node.text:
                                # Fallback to simple text node (might contain address and ZIP)
                                location_address = civic_text_node.text.strip()
                                logger.debug(f"Extracted address/text from <civicAddressText>: {location_address}")
                                # Try to extract ZIP from this text if not found above (simple regex could work)
                                if not zip_code and location_address:
                                    import re
                                    zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', location_address) # Find 5 or 9 digit zip
                                    if zip_match:
                                        zip_code = zip_match.group(1)
                                        logger.debug(f"Extracted ZIP '{zip_code}' from civicAddressText using regex.")

                        except ET.ParseError as xml_e: logger.error(f"Failed parsing locationByValue XML for {loc_comp_id_for_log}: {xml_e}")
                    else: # Try parsing directly if not our wrapper
                         # (Keep original XML parsing logic here as fallback)
                         pass # Simplified for brevity

                except Exception as e: logger.error(f"Unexpected error during XML location parsing for {loc_comp_id_for_log}: {e}", exc_info=True)

            # --- Handle locationByReferenceUrl (less common for coords) ---
            elif loc_ref_url and isinstance(loc_ref_url, str) and not location_address and not location_coords:
                 logger.info(f"LocationComponent {loc_comp_id_for_log} has reference URL: {loc_ref_url}. Geocoding might fail.")
                 location_address = f"Reference URL: {loc_ref_url}" # Use as pseudo-address

            # --- Geocode if coordinates are MISSING but address IS present ---
            if location_address and not location_coords:
                logger.info(f"Coordinates missing for {loc_comp_id_for_log}. Attempting to geocode address: '{location_address}'")
                try:
                    # Ensure geocoding service is called
                    location_coords = get_coordinates(location_address)
                    if location_coords:
                        logger.info(f"Geocoding successful for '{location_address}': {location_coords}")
                    else:
                        logger.warning(f"Geocoding failed for address: '{location_address}'")
                except Exception as geo_e:
                    logger.error(f"Error during geocoding call for '{location_address}': {geo_e}", exc_info=True)
                    location_coords = None # Ensure it's None on error
            elif not location_address and not location_coords:
                 logger.warning(f"No address or coordinates found/extracted for LocationComponent {loc_comp_id_for_log}.")
            elif location_coords:
                 logger.debug(f"Using existing coordinates for {loc_comp_id_for_log}: {location_coords}")


        # --- Extract Source Agency Name (using agency_ref_id) ---
        source_agency_name = "Unknown Source"
        agency_components = eido_dict.get('agencyComponent', [])
        if agency_ref_id and isinstance(agency_components, list):
            agency_found = False
            # --- >> ADDED: Check item type inside loop << ---
            for agency_comp in agency_components:
                 if isinstance(agency_comp, dict): # Process only dicts
                    # Check agencyIdentifier and the common '$id' pattern
                    comp_id = agency_comp.get('agencyIdentifier', agency_comp.get('$id'))
                    if comp_id == agency_ref_id:
                        # Prefer agencyName, fallback to ID, ensure it's a string
                        agency_name_val = agency_comp.get('agencyName', agency_ref_id)
                        source_agency_name = str(agency_name_val) if agency_name_val is not None else agency_ref_id
                        agency_found = True
                        logger.debug(f"Found source agency: {source_agency_name} (ID: {comp_id})")
                        break
                 else:
                     logger.warning(f"Msg {message_id}: Found non-dictionary item in agencyComponent: {type(agency_comp)}. Skipping.")
            # --- >> END ADDED << ---
            if not agency_found:
                 logger.warning(f"Referenced Agency ID '{agency_ref_id}' not found in agencyComponent list. Using ID as source name.")
                 source_agency_name = agency_ref_id # Fallback to the ID itself
        elif agency_ref_id: # If agency_ref_id came from sendingSystemIdentifier or similar
             source_agency_name = agency_ref_id
             logger.debug(f"Using agency reference ID '{agency_ref_id}' directly as source name (not found in components).")


        # --- Create ReportCoreData ---
        try:
            core_data = ReportCoreData(
                # report_id generated automatically
                external_incident_id=incident_tracking_id, # Use extracted ID
                timestamp=timestamp,
                incident_type=incident_type,
                description=full_description,
                location_address=location_address, # Use extracted/formatted address
                coordinates=location_coords,       # Use extracted OR geocoded coords
                zip_code=zip_code,                 # Use extracted zip
                source=source_agency_name,         # Use extracted agency name
                original_document_id=message_id,
                original_eido_dict=eido_dict
            )
            logger.info(f"Successfully extracted core data for Report {core_data.report_id[:8]} (Coords: {core_data.coordinates}, ZIP: {core_data.zip_code})")
            return core_data
        except Exception as pydantic_error:
             logger.error(f"Failed to create ReportCoreData instance for Msg {message_id}: {pydantic_error}", exc_info=True)
             return None



    def process_report_json(self, json_data: Dict) -> Dict:
        """
        Processes a single EIDO report provided as a JSON dictionary.
        Extracts data directly from dict, matches, stores, and calls LLM.
        Returns a dictionary summarizing the processing result.
        """
        # Use .get() for safer access, provide fallback ID
        message_id = json_data.get('eidoMessageIdentifier', json_data.get('$id', f"unknown_{str(uuid.uuid4())[:8]}"))
        logger.info(f"--- Processing EIDO Message Dict ID: {message_id} ---")

        # 1. Validate Input Type (ensure it's a dictionary)
        if not isinstance(json_data, dict):
            logger.error(f"Input data for message ID '{message_id}' is not a dictionary (Type: {type(json_data)}). Skipping.")
            return {
                "status": "Input Error: Data must be a JSON object (dictionary).",
                "message_id": message_id,
                "incident_id": None,
                "is_new_incident": False,
                "summary": None,
                "actions": None
            }

        # 2. Extract Core Information (Directly from Dict)
        try:
            # Call the modified extraction function
            core_data = self._extract_core_data_from_dict(json_data)
            if not core_data:
                # Error already logged in _extract_core_data_from_dict
                return {
                    "status": "Failed processing: Could not extract core data (check logs for details).",
                    "message_id": message_id,
                    "incident_id": None,
                    "is_new_incident": False,
                    "summary": None,
                    "actions": None
                }
        except Exception as e:
            # Catch unexpected errors during extraction itself
            logger.error(f"Msg '{message_id}': Unexpected error during core data extraction: {e}", exc_info=True)
            return {
                 # --- Return the specific error type ---
                 "status": f"Processing Error: Core data extraction failed ({type(e).__name__})",
                 # --- End change ---
                 "message_id": message_id,
                 "incident_id": None,
                 "is_new_incident": False,
                 "summary": None,
                 "actions": None
            }

        # 3. Generate Embedding (Optional, based on extracted description)
        description_embedding = None
        if core_data.description and "No description" not in core_data.description:
            try:
                description_embedding = generate_embedding(core_data.description)
                # Log embedding status (debug/info)
            except Exception as e:
                logger.warning(f"Msg '{message_id}': Failed to generate embedding: {e}")

        # 4. Match to Existing Incident or Create New
        matched_incident = None
        matched_incident_id = None
        match_score = 0.0
        match_reason = "Matching not attempted or failed"
        try:
            active_incidents = incident_store.get_active_incidents()
            logger.debug(f"Msg '{message_id}': Found {len(active_incidents)} active incidents for matching.")

            matched_incident_id, match_score, match_reason = find_match_for_report(core_data, active_incidents)

            if matched_incident_id:
                matched_incident = incident_store.get_incident(matched_incident_id)
                if not matched_incident:
                     logger.error(f"Msg '{message_id}': Matching returned ID {matched_incident_id[:8]} but incident not found! Treating as new.")
                     matched_incident_id = None # Reset

        except Exception as e:
             logger.error(f"Msg '{message_id}': Error during incident matching phase: {e}", exc_info=True)
             matched_incident = None
             matched_incident_id = None
             match_reason = f"Matching Error: {type(e).__name__}"

        # 5. Update Store and Prepare for LLM
        incident_to_process = None
        is_new_incident = False
        incident_id = None # Initialize

        if matched_incident:
            incident_id = matched_incident.incident_id
            logger.info(f"Msg '{message_id}': Matched to existing Incident {incident_id[:8]} (Score: {match_score:.2f}, Reason: {match_reason}).")
            try:
                 match_info_str = f"Matched Report {core_data.report_id[:8]} (Score: {match_score:.2f}, Reason: {match_reason})"
                 matched_incident.add_report_core_data(core_data, match_info=match_info_str)
                 # Add logic here if status needs update based on EIDO status text (e.g., primary_comp.get('incidentStatusCommonRegistryText'))
            except Exception as e:
                 logger.error(f"Failed adding report {core_data.report_id[:8]} to incident {incident_id[:8]}: {e}", exc_info=True)

            incident_to_process = matched_incident
            is_new_incident = False
        else:
            logger.info(f"Msg '{message_id}': No match found or error during matching (Reason: {match_reason}). Creating new Incident.")
            try:
                # Create a new Incident object
                incident_to_process = Incident(
                    incident_type=core_data.incident_type,
                    status="Active" # Initial status
                )
                # Add the first report's data
                match_info_str = f"Created from Report {core_data.report_id[:8]} (Reason: {match_reason})"
                incident_to_process.add_report_core_data(core_data, match_info=match_info_str)
                incident_id = incident_to_process.incident_id # Get the generated ID
                logger.info(f"Created new Incident {incident_id[:8]} from Report {core_data.report_id[:8]}.")
                is_new_incident = True
            except Exception as e:
                 logger.error(f"Msg '{message_id}': Failed to create new Incident object or add initial data: {e}", exc_info=True)
                 # Cannot proceed without an incident object
                 return {
                     "status": f"Processing Error: Failed to initialize new incident ({type(e).__name__})",
                     "message_id": message_id,
                     "incident_id": None,
                     "is_new_incident": True, # It was intended to be new
                     "summary": None,
                     "actions": None
                 }

        # 6. Generate Summary and Recommendations using LLM (if incident_to_process exists)
        if incident_to_process:
            try:
                history = ""
                if not is_new_incident and len(incident_to_process.reports_core_data) > 1:
                    history = incident_to_process.get_full_description_history(exclude_latest=True)

                # Call LLM for summary
                new_summary = summarize_incident(history, core_data)
                if new_summary:
                    incident_to_process.summary = new_summary
                    logger.debug(f"Incident {incident_id[:8]}: Generated new summary.")
                else:
                    logger.error(f"Incident {incident_id[:8]}: Failed to generate summary (LLM call returned None or error).")
                    # Keep previous summary or use a placeholder if it's the first time
                    if not incident_to_process.summary or "not yet generated" in incident_to_process.summary:
                         incident_to_process.summary = f"LLM Error: Could not generate summary. Last report description: {core_data.description}"

                # Call LLM for recommendations
                new_recommendations = recommend_actions(incident_to_process.summary, core_data)
                if new_recommendations:
                    incident_to_process.recommended_actions = new_recommendations
                    logger.debug(f"Incident {incident_id[:8]}: Generated new recommendations.")
                else:
                    logger.error(f"Incident {incident_id[:8]}: Failed to generate recommendations (LLM call returned None or error).")
                    if not incident_to_process.recommended_actions:
                         incident_to_process.recommended_actions = ["LLM Error: Could not generate recommendations."]

            except Exception as e:
                logger.error(f"Incident {incident_id[:8]}: Error during LLM interaction: {e}", exc_info=True)
                incident_to_process.summary = f"Error during LLM processing: {type(e).__name__}"
                incident_to_process.recommended_actions = [f"Error during LLM processing: {type(e).__name__}"]
        else:
             # Should not happen if creation/matching logic is correct
             logger.critical(f"Msg '{message_id}': incident_to_process object was None before LLM step. This indicates a critical logic error.")
             return {
                 "status": "Processing Error: Internal state error before LLM.",
                 "message_id": message_id,
                 "incident_id": incident_id, # May be None
                 "is_new_incident": is_new_incident,
                 "summary": None,
                 "actions": None
             }

        # 7. Save the incident back to the store
        try:
            incident_store.save_incident(incident_to_process)
            logger.info(f"Msg '{message_id}': Successfully processed. Report {core_data.report_id[:8]} added/updated Incident {incident_id[:8]}.")
            return {
                "status": "Success",
                "message_id": message_id, # Return the original EIDO message ID (or generated if needed)
                "incident_id": incident_id,
                "is_new_incident": is_new_incident,
                "summary": incident_to_process.summary,
                "actions": incident_to_process.recommended_actions
            }
        except Exception as e:
             logger.critical(f"CRITICAL: Failed to save incident {incident_id[:8]} to store! Data might be inconsistent. Error: {e}", exc_info=True)
             return {
                 "status": f"Processing Error: Failed to save incident to store ({type(e).__name__})",
                 "message_id": message_id,
                 "incident_id": incident_id,
                 "is_new_incident": is_new_incident,
                 "summary": incident_to_process.summary, # Return potentially updated data
                 "actions": incident_to_process.recommended_actions
             }

    # --- NEW METHOD ---
    def process_alert_text(self, alert_text: str) -> List[Dict]:
        """ Processes raw alert text, potentially containing multiple events. """
        logger.info("--- Processing Raw Alert Text Block ---")
        results = []
        if not alert_text or not isinstance(alert_text, str):
             logger.error("Invalid alert text input.")
             return [{"status": "Input Error: Alert text cannot be empty.", "message_id": "N/A", "incident_id": None}]

        # --- NEW: Check if input looks like JSON ---
        cleaned_text = alert_text.strip()
        if (cleaned_text.startswith('{') and cleaned_text.endswith('}')) or \
           (cleaned_text.startswith('[') and cleaned_text.endswith(']')):
            logger.warning("Input provided to 'Raw Alert Text' processing looks like JSON.")
            logger.warning("Attempting to process, but for JSON input, please use the 'EIDO JSON' tab/endpoint.")
            # Optional: Return an error immediately?
            # return [{"status": "Input Error: Raw text input appears to be JSON. Use the correct input method.", "message_id": "N/A", "incident_id": None}]
            # Or let it proceed, but the splitter might behave unexpectedly (as seen in logs)
        # --- END NEW CHECK ---


        # 1. Split text into potential events using LLM
        event_texts: Optional[List[str]] = None
        try:
            event_texts = split_raw_text_into_events(alert_text)
            # ... (keep logging added previously) ...
            if event_texts is None: logger.warning("LLM splitting function returned None.")
            elif len(event_texts) == 1: logger.info("LLM splitting function returned a single event.")
            else: logger.info(f"LLM splitting function returned {len(event_texts)} potential events.")
        except Exception as e:
             logger.error(f"Critical error during event splitting call: {e}", exc_info=True)
             event_texts = None

        # Fallback logic
        if not event_texts:
            logger.warning("Splitting failed or yielded no events. Processing entire block as one event.")
            event_texts = [alert_text]

        logger.info(f"Attempting to process {len(event_texts)} event text(s).")

        # 2. Process each potential event text
        # ... (Keep the rest of the loop as is - calling parse_alert_to_eido_dict and process_report_json) ...
        for i, single_event_text in enumerate(event_texts):
            # ... (process single_event_text) ...
            if not single_event_text.strip(): # Skip empty
                logger.warning(f"Skipping empty event text #{i+1}.")
                results.append({"status": f"Processing Error: Event text #{i+1} was empty.", "message_id": f"text_event_{i+1}_empty", "incident_id": None})
                continue
            generated_eido_dict = parse_alert_to_eido_dict(single_event_text)
            if not generated_eido_dict: # Handle parsing failure
                 logger.error(f"Failed to parse event text #{i+1}.")
                 results.append({"status": f"Processing Error: Failed to parse event text #{i+1}.", "message_id": f"text_event_{i+1}_parse_fail", "incident_id": None, "original_text_snippet": single_event_text[:100] + "..."})
                 continue
            try: # Process the dict
                result_dict = self.process_report_json(generated_eido_dict)
                result_dict['source_event_index'] = i + 1
                result_dict['source_event_total'] = len(event_texts)
                results.append(result_dict)
            except Exception as e: # Handle processing failure
                 logger.error(f"Error processing dictionary for event text #{i+1}: {e}", exc_info=True)
                 results.append({"status": f"Processing Error: Failed during generated dict processing for event #{i+1} ({type(e).__name__})", "message_id": generated_eido_dict.get('eidoMessageIdentifier', f'llm_parsed_{i+1}_err'), "incident_id": None})

        # ... (Keep final return logic) ...
        return results


# --- Singleton Instance ---
eido_agent_instance = EidoAgent()