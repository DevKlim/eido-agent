# agent/agent_core.py
import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
import uuid
import xml.etree.ElementTree as ET # Import XML parser

# Use the simplified schemas
try:
    from data_models.schemas import ReportCoreData, Incident # Removed EidoMessage import
    from services.storage import incident_store
    from services.geocoding import get_coordinates
    from services.embedding import generate_embedding
    from agent.matching import find_match_for_report
    from agent.llm_interface import summarize_incident, recommend_actions, extract_eido_from_alert_text # Import new LLM func if needed directly
    from agent.alert_parser import parse_alert_to_eido_dict # <-- Import the new parser function
except ImportError as e:
    print(f"CRITICAL ERROR in agent_core.py: Failed to import dependencies - {e}")
    raise SystemExit(f"Agent Core import failed: {e}") from e

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
        if not primary_comp: # This now correctly checks if primary_comp is still None
             logger.error(f"Msg {message_id}: Cannot extract core data - No valid IncidentComponent or CallComponent dictionary found.")
             return None

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
        loc_ref_input = primary_comp.get('locationReference')
        loc_ref_id = self._resolve_ref_string_from_dict(loc_ref_input)

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


        # --- Extract Location Details (using loc_ref_id) ---
        location_address: Optional[str] = None
        location_coords: Optional[Tuple[float, float]] = None
        primary_location_comp = None
        location_components = eido_dict.get('locationComponent', [])

        if isinstance(location_components, list): # Ensure it's a list before proceeding
            if loc_ref_id:
                # --- >> ADDED: Check item type inside loop << ---
                for loc_comp in location_components:
                    if isinstance(loc_comp, dict): # Process only dicts
                        # Check both componentIdentifier and the common '$id' pattern
                        comp_id = loc_comp.get('componentIdentifier', loc_comp.get('$id'))
                        if comp_id == loc_ref_id:
                            primary_location_comp = loc_comp
                            logger.debug(f"Found referenced LocationComponent by ID: {comp_id}")
                            break
                    else:
                         logger.warning(f"Msg {message_id}: Found non-dictionary item in locationComponent: {type(loc_comp)}. Skipping.")
                # --- >> END ADDED << ---
                if not primary_location_comp:
                     logger.warning(f"Referenced Location ID '{loc_ref_id}' not found in locationComponent list.")
            elif location_components: # If no ref ID, try the first valid dict in the list
                 # --- >> ADDED: Find first valid dict << ---
                 for loc_comp in location_components:
                     if isinstance(loc_comp, dict):
                         primary_location_comp = loc_comp
                         logger.warning(f"Primary component missing locationReference. Using first valid LocationComponent as fallback.")
                         break
                     else:
                          logger.warning(f"Msg {message_id}: Found non-dictionary item in locationComponent: {type(loc_comp)}. Skipping.")
                 # --- >> END ADDED << ---
                 if not primary_location_comp:
                      logger.warning(f"No valid dictionary found in locationComponent list to use as fallback.")

        else:
             logger.debug(f"No LocationComponent list found or it's not a list in message {message_id}.")

        # Extract details only if a valid primary_location_comp (dict) was found
        if primary_location_comp: # No need for isinstance check here, logic above ensures it's a dict or None
            loc_comp_id_for_log = primary_location_comp.get('componentIdentifier', primary_location_comp.get('$id', 'N/A'))

            # --- Handle locationByValue (Assume XML String for now) ---
            loc_val = primary_location_comp.get('locationByValue')
            # --- >> REFINED CHECK: Ensure loc_val is a non-empty string before stripping/parsing << ---
            if loc_val and isinstance(loc_val, str):
                loc_val_stripped = loc_val.strip()
                if loc_val_stripped.startswith('<?xml'):
                    logger.debug(f"Attempting basic XML parse for locationByValue in {loc_comp_id_for_log}")
                    try:
                        # (Keep existing XML parsing logic)
                        namespaces = { # Common namespaces, add more if needed
                            'ca': 'urn:ietf:params:xml:ns:pidf:geopriv10:civicAddr',
                            'gml': 'http://www.opengis.net/gml'
                        }
                        root = ET.fromstring(loc_val_stripped) # Use stripped version
                        # Try finding civic address elements
                        civic_node = root.find('.//ca:civicAddress', namespaces)
                        if civic_node is not None:
                            addr_parts = [
                                civic_node.findtext('ca:HNO', None, namespaces), civic_node.findtext('ca:HNS', None, namespaces),
                                civic_node.findtext('ca:PRD', None, namespaces), civic_node.findtext('ca:RD', None, namespaces),
                                civic_node.findtext('ca:STS', None, namespaces), civic_node.findtext('ca:POD', None, namespaces),
                                civic_node.findtext('ca:A6', None, namespaces), civic_node.findtext('ca:A3', None, namespaces), # City
                                civic_node.findtext('ca:A1', None, namespaces), # State
                                civic_node.findtext('ca:PC', None, namespaces) # Zip
                            ]
                            location_address = " ".join(filter(None, addr_parts)).strip()
                            logger.debug(f"Extracted address from XML: {location_address}")

                        # Try finding geodetic coordinates (GML pos)
                        pos_node = root.find('.//gml:pos', namespaces)
                        if pos_node is not None and pos_node.text:
                            coords_text = pos_node.text.strip().split()
                            if len(coords_text) >= 2:
                                try:
                                    lat = float(coords_text[0])
                                    lon = float(coords_text[1])
                                    location_coords = (lat, lon)
                                    logger.debug(f"Extracted coordinates from XML <gml:pos>: {location_coords}")
                                except (ValueError, TypeError):
                                    logger.warning(f"Could not parse coordinates from XML <gml:pos> text: '{pos_node.text}'")
                    except ET.ParseError as xml_e:
                        logger.error(f"Failed to parse XML location for {loc_comp_id_for_log}: {xml_e}")
                    except Exception as e:
                         logger.error(f"Unexpected error during XML location parsing for {loc_comp_id_for_log}: {e}", exc_info=True)
                # else: # Optional: Handle non-XML string if needed
                #    logger.debug(f"locationByValue is a non-XML string: '{loc_val_stripped[:50]}...'")
                #    if not location_address: location_address = loc_val_stripped # Use as address if not already found
            # --- >> END REFINED CHECK << ---

            # --- Handle locationByReferenceUrl ---
            loc_ref_url = primary_location_comp.get('locationReferenceUrl')
            if loc_ref_url and isinstance(loc_ref_url, str) and not location_address and not location_coords:
                 logger.info(f"LocationComponent {loc_comp_id_for_log} has reference URL: {loc_ref_url}. Geocoding might fail.")
                 location_address = f"Reference URL: {loc_ref_url}" # Use URL as pseudo-address

            # --- Geocode if coordinates are missing but address is present ---
            if location_address and isinstance(location_address, str) and not location_coords:
                logger.debug(f"Attempting to geocode address: '{location_address}'")
                location_coords = get_coordinates(location_address)


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
                external_incident_id=incident_tracking_id,
                timestamp=timestamp,
                incident_type=incident_type,
                description=full_description,
                location_address=location_address,
                coordinates=location_coords,
                source=source_agency_name,
                original_document_id=message_id,
                original_eido_dict=eido_dict
            )
            logger.info(f"Successfully extracted core data for Report {core_data.report_id[:8]} (ExtIncidentID: {incident_tracking_id}, Type: {incident_type})")
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
    def process_alert_text(self, alert_text: str) -> Dict:
        """
        Processes raw alert text by parsing it into an EIDO-like dictionary
        using an LLM, and then feeding it into the standard JSON processing pipeline.

        Args:
            alert_text: The raw text of the alert.

        Returns:
            A dictionary summarizing the processing result, similar to process_report_json.
        """
        logger.info("--- Processing Raw Alert Text ---")
        if not alert_text or not isinstance(alert_text, str):
             logger.error("Received invalid alert text input.")
             return {
                "status": "Input Error: Alert text cannot be empty.",
                "message_id": "N/A", # No message ID from raw text initially
                "incident_id": None,
                "is_new_incident": False,
                "summary": None,
                "actions": None
            }

        # 1. Parse alert text into EIDO-like dictionary
        generated_eido_dict = None # Initialize
        try:
            generated_eido_dict = parse_alert_to_eido_dict(alert_text)
        except Exception as e:
             logger.error(f"Critical error during alert parsing function call: {e}", exc_info=True)
             # generated_eido_dict remains None

        if not generated_eido_dict:
            logger.error("Failed to parse alert text into EIDO-like structure (parse_alert_to_eido_dict returned None or raised error).")
            return {
                "status": "Processing Error: Failed to parse alert text using LLM.",
                "message_id": "N/A",
                "incident_id": None,
                "is_new_incident": False,
                "summary": None,
                "actions": None
            }

        # 2. Process the generated dictionary using the existing pipeline
        logger.info("Handing off generated EIDO-like dictionary to JSON processing pipeline.")
        try:
            # The generated dict contains a message ID ('llm_parsed_...')
            return self.process_report_json(generated_eido_dict)
        except Exception as e:
             # Catch errors specifically from the process_report_json call
             logger.error(f"Error processing the dictionary generated from alert text: {e}", exc_info=True)
             return {
                 "status": f"Processing Error: Failed during EIDO-like dict processing ({type(e).__name__})",
                 "message_id": generated_eido_dict.get('eidoMessageIdentifier', 'llm_parsed_N/A'), # Try to get ID from generated dict
                 "incident_id": None,
                 "is_new_incident": False, # Unknown at this point
                 "summary": None,
                 "actions": None
             }


# --- Singleton Instance ---
eido_agent_instance = EidoAgent()