import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any, Union
import uuid
import xml.etree.ElementTree as ET # Import XML parser
import re # For ZIP code extraction

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
                    if not current: return default
                    current = current[0]
                    if not isinstance(current, dict): return default
                if not isinstance(current, dict): return default
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
        return ref_id

    def _extract_core_data_from_dict(self, eido_dict: Dict) -> Optional[ReportCoreData]:
        """
        Extracts key information directly from an EIDO message dictionary
        into the simplified ReportCoreData structure.
        Uses safe access methods (.get) to handle missing fields gracefully.
        """
        if not isinstance(eido_dict, dict):
            logger.error(f"Input to _extract_core_data_from_dict is not a dictionary (Type: {type(eido_dict)}). Cannot extract.")
            return None

        message_id = eido_dict.get('eidoMessageIdentifier', eido_dict.get('$id'))
        if not message_id:
            message_id = f"unknown_{str(uuid.uuid4())[:8]}"
            logger.warning(f"EIDO message dictionary missing 'eidoMessageIdentifier' and '$id'. Assigned temporary ID: {message_id}")
        logger.debug(f"Extracting core data from EIDO Message Dict: {message_id}")

        incident_comp_list = eido_dict.get('incidentComponent')
        call_comp_list = eido_dict.get('callComponent')
        primary_comp = None
        source_component_type = "Unknown"

        if isinstance(incident_comp_list, list):
            for comp in incident_comp_list:
                if isinstance(comp, dict):
                    primary_comp = comp
                    source_component_type = "IncidentComponent"
                    if len(incident_comp_list) > 1:
                         logger.warning(f"Msg {message_id}: Multiple IncidentComponents found ({len(incident_comp_list)}). Using first valid dictionary.")
                    break
        if not primary_comp and isinstance(call_comp_list, list):
             for comp in call_comp_list:
                 if isinstance(comp, dict):
                     primary_comp = comp
                     source_component_type = "CallComponent"
                     logger.warning(f"Msg {message_id}: No valid IncidentComponent found. Using CallComponent for core data.")
                     if len(call_comp_list) > 1:
                          logger.warning(f"Msg {message_id}: Multiple CallComponents found ({len(call_comp_list)}). Using first valid dictionary.")
                     break
        if not primary_comp:
            logger.error(f"Msg {message_id}: No suitable primary component (IncidentComponent or CallComponent dictionary) found for data extraction.")
            return None

        incident_tracking_id = primary_comp.get('incidentTrackingIdentifier')
        if not incident_tracking_id and source_component_type == "CallComponent":
             incident_tracking_id = primary_comp.get('callTrackingIdentifier')

        timestamp_str = primary_comp.get('lastUpdateTimeStamp', eido_dict.get('lastUpdateTimeStamp'))
        timestamp = ReportCoreData.ensure_timezone_aware(timestamp_str or datetime.now(timezone.utc))

        raw_incident_type = primary_comp.get('incidentTypeCommonRegistryText')
        if not raw_incident_type and source_component_type == "CallComponent":
             raw_incident_type = primary_comp.get('callTypeCommonRegistryText', "Inferred Call")
        incident_type = None
        if isinstance(raw_incident_type, list) and raw_incident_type:
            first_type = raw_incident_type[0]
            incident_type = str(first_type) if first_type is not None else "Unknown"
        elif isinstance(raw_incident_type, str):
            incident_type = raw_incident_type
        else:
            incident_type = "Unknown"

        loc_ref_id = self._resolve_ref_string_from_dict(primary_comp.get('locationReference'))
        
        agency_ref_input = primary_comp.get('incidentOriginatingAgencyIdentifier')
        if not agency_ref_input: agency_ref_input = primary_comp.get('updatedByAgencyReference')
        agency_ref_id = self._resolve_ref_string_from_dict(agency_ref_input)
        if not agency_ref_id:
             agency_ref_id = eido_dict.get('sendingSystemIdentifier')
             if agency_ref_id: logger.debug(f"Using sendingSystemIdentifier '{agency_ref_id}' as fallback source agency ID.")

        descriptions = []
        notes_list = eido_dict.get('notesComponent', [])
        if isinstance(notes_list, list):
             for note_comp in notes_list:
                 if isinstance(note_comp, dict):
                     note_text = note_comp.get('noteText')
                     if note_text:
                         note_ts_str = note_comp.get('noteDateTimeStamp')
                         note_ts = ReportCoreData.ensure_timezone_aware(note_ts_str or timestamp)
                         timestamp_str_fmt = note_ts.strftime('%Y-%m-%d %H:%M:%S Z')
                         author_ref_id = self._resolve_ref_string_from_dict(note_comp.get('authorReference'))
                         author_info = f" (AuthorRef: {author_ref_id})" if author_ref_id else ""
                         descriptions.append(f"[{timestamp_str_fmt}]{author_info}: {note_text}")
                 else:
                     logger.warning(f"Msg {message_id}: Found non-dictionary item in notesComponent: {type(note_comp)}. Skipping.")
        
        comments_list = eido_dict.get('commentsComponent', [])
        if isinstance(comments_list, list):
            for comment_comp in comments_list:
                if isinstance(comment_comp, dict):
                    comment_text = comment_comp.get('commentText') 
                    if comment_text:
                        comment_ts_str = comment_comp.get('commentDateTimeStamp')
                        comment_ts = ReportCoreData.ensure_timezone_aware(comment_ts_str or timestamp)
                        timestamp_str_fmt = comment_ts.strftime('%Y-%m-%d %H:%M:%S Z')
                        author_ref_id = self._resolve_ref_string_from_dict(comment_comp.get('authorReference'))
                        author_info = f" (AuthorRef: {author_ref_id})" if author_ref_id else ""
                        descriptions.append(f"[Comment @ {timestamp_str_fmt}]{author_info}: {comment_text}")
                else:
                    logger.warning(f"Msg {message_id}: Found non-dictionary item in commentsComponent: {type(comment_comp)}. Skipping.")

        full_description = "\n".join(descriptions) if descriptions else primary_comp.get('incidentSummaryText')
        if not full_description: full_description = "No description, notes, or comments found."

        location_address: Optional[str] = None
        location_coords: Optional[Tuple[float, float]] = None
        zip_code: Optional[str] = None
        primary_location_comp_dict = None
        location_components = eido_dict.get('locationComponent', [])

        if isinstance(location_components, list):
            found_loc = False
            for loc_comp_item in location_components:
                if isinstance(loc_comp_item, dict):
                    comp_id = loc_comp_item.get('componentIdentifier', loc_comp_item.get('$id'))
                    if loc_ref_id and comp_id == loc_ref_id:
                        primary_location_comp_dict = loc_comp_item; found_loc = True; break
            if not found_loc: 
                 for loc_comp_item in location_components:
                     if isinstance(loc_comp_item, dict):
                         primary_location_comp_dict = loc_comp_item; break
        
        if primary_location_comp_dict:
            loc_comp_id_for_log = primary_location_comp_dict.get('componentIdentifier', primary_location_comp_dict.get('$id', 'N/A'))
            logger.debug(f"Processing LocationComponent: {loc_comp_id_for_log}")
            loc_val = primary_location_comp_dict.get('locationByValue')
            loc_ref_url = primary_location_comp_dict.get('locationReferenceUrl')

            if loc_val and isinstance(loc_val, str) and loc_val.strip().startswith('<?xml'):
                logger.debug(f"Attempting XML parse for locationByValue in {loc_comp_id_for_log}")
                try:
                    namespaces = {'ca': 'urn:ietf:params:xml:ns:pidf:geopriv10:civicAddr', 'gml': 'http://www.opengis.net/gml'}
                    root_str = loc_val.strip()
                    
                    xml_root_node = None
                    try:
                        xml_root_node = ET.fromstring(root_str)
                    except ET.ParseError as xml_e_direct:
                        logger.debug(f"Direct XML parse failed for locationByValue '{loc_comp_id_for_log}': {xml_e_direct}. Checking for <location> wrapper...")
                        if root_str.startswith('<location>') and root_str.endswith('</location>'):
                             inner_xml_str = root_str[len('<location>'):-len('</location>')].strip()
                             if inner_xml_str.startswith('<?xml'): # If inner content is a full XML doc itself
                                inner_xml_str = inner_xml_str[inner_xml_str.find('?>')+2:].strip() # Remove XML declaration
                             try:
                                 xml_root_node = ET.fromstring(inner_xml_str)
                                 logger.debug("Successfully parsed after removing <location> wrapper and potentially inner XML declaration.")
                             except ET.ParseError as xml_e_inner:
                                 logger.error(f"Failed parsing inner XML of locationByValue for {loc_comp_id_for_log}: {xml_e_inner}")
                        else:
                            logger.warning(f"XML in locationByValue for {loc_comp_id_for_log} is not wrapped in <location> and failed direct parse.")


                    if xml_root_node is not None:
                        # If the root itself is <location>, look for its children directly
                        target_search_node = xml_root_node
                        if xml_root_node.tag.lower() == 'location': 
                            logger.debug("XML root is <location>, searching children for GML/Civic data.")
                        
                        gml_point = target_search_node.find('.//gml:Point', namespaces)
                        civic_text_node = target_search_node.find('.//civicAddressText') 
                        ca_node = target_search_node.find('.//ca:civicAddress', namespaces)

                        if gml_point is not None:
                            pos_node = gml_point.find('.//gml:pos', namespaces)
                            if pos_node is not None and pos_node.text:
                                coords_text = pos_node.text.strip().split()
                                if len(coords_text) >= 2:
                                    try: location_coords = (float(coords_text[0]), float(coords_text[1])); logger.debug(f"Extracted coords from <gml:pos>: {location_coords}")
                                    except (ValueError, TypeError): logger.warning(f"Could not parse coords from <gml:pos>: '{pos_node.text}'")
                        
                        extracted_addr_parts = []
                        if ca_node is not None:
                            addr_fields_order = ['HNO', 'HNS', 'PRD', 'RD', 'STS', 'POD', 'A6', 'A3', 'A1', 'PC']
                            for tag in addr_fields_order:
                                 el_text = ca_node.findtext(f'ca:{tag}', None, namespaces)
                                 if el_text: extracted_addr_parts.append(el_text)
                            location_address = " ".join(filter(None, extracted_addr_parts)).strip()
                            zip_code = ca_node.findtext('ca:PC', None, namespaces)
                            logger.debug(f"Extracted address from <ca:civicAddress>: {location_address}, ZIP: {zip_code}")
                        
                        if not location_address and civic_text_node is not None and civic_text_node.text:
                            location_address = civic_text_node.text.strip()
                            logger.debug(f"Extracted address/text from <civicAddressText>: {location_address}")
                        
                        if not zip_code and location_address: 
                            zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', location_address)
                            if zip_match: zip_code = zip_match.group(1); logger.debug(f"Extracted ZIP '{zip_code}' from address text using regex.")

                except Exception as e: logger.error(f"Unexpected error during XML location parsing for {loc_comp_id_for_log}: {e}", exc_info=True)
            
            elif loc_val and isinstance(loc_val, str) and not location_address: 
                 location_address = loc_val.strip()
                 logger.debug(f"Using non-XML locationByValue as address: '{location_address}'")
                 if not zip_code: 
                     zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', location_address)
                     if zip_match: zip_code = zip_match.group(1); logger.debug(f"Extracted ZIP '{zip_code}' from non-XML locationByValue using regex.")

            elif loc_ref_url and isinstance(loc_ref_url, str) and not location_address and not location_coords:
                 logger.info(f"LocationComponent {loc_comp_id_for_log} has reference URL: {loc_ref_url}. Geocoding might fail if this is the only info.")
                 location_address = f"Reference URL: {loc_ref_url}"

            if location_address and not location_coords:
                logger.info(f"Coordinates missing for {loc_comp_id_for_log}. Attempting to geocode address: '{location_address}'")
                try:
                    location_coords = get_coordinates(location_address)
                    if location_coords: logger.info(f"Geocoding successful for '{location_address}': {location_coords}")
                    else: logger.warning(f"Geocoding failed for address: '{location_address}'")
                except Exception as geo_e: logger.error(f"Error during geocoding call for '{location_address}': {geo_e}", exc_info=True); location_coords = None
            elif not location_address and not location_coords:
                 logger.warning(f"No address or coordinates found/extracted for LocationComponent {loc_comp_id_for_log}.")
            elif location_coords:
                 logger.debug(f"Using existing coordinates for {loc_comp_id_for_log}: {location_coords}")
        else: 
            logger.warning(f"Msg {message_id}: No valid LocationComponent found or referenced.")


        source_agency_name = "Unknown Source"
        agency_components = eido_dict.get('agencyComponent', [])
        if agency_ref_id and isinstance(agency_components, list):
            agency_found = False
            for agency_comp in agency_components:
                 if isinstance(agency_comp, dict):
                    comp_id = agency_comp.get('agencyIdentifier', agency_comp.get('$id'))
                    if comp_id == agency_ref_id:
                        agency_name_val = agency_comp.get('agencyName', agency_ref_id)
                        source_agency_name = str(agency_name_val) if agency_name_val is not None else agency_ref_id
                        agency_found = True
                        logger.debug(f"Found source agency: {source_agency_name} (ID: {comp_id})")
                        break
                 else:
                     logger.warning(f"Msg {message_id}: Found non-dictionary item in agencyComponent: {type(agency_comp)}. Skipping.")
            if not agency_found:
                 logger.warning(f"Referenced Agency ID '{agency_ref_id}' not found in agencyComponent list. Using ID as source name.")
                 source_agency_name = agency_ref_id
        elif agency_ref_id:
             source_agency_name = agency_ref_id
             logger.debug(f"Using agency reference ID '{agency_ref_id}' directly as source name (not found in components).")
        elif eido_dict.get('sendingSystemIdentifier'): 
            source_agency_name = eido_dict['sendingSystemIdentifier']
            logger.debug(f"Using top-level sendingSystemIdentifier '{source_agency_name}' as source agency name.")


        try:
            core_data = ReportCoreData(
                external_incident_id=incident_tracking_id,
                timestamp=timestamp,
                incident_type=incident_type,
                description=full_description,
                location_address=location_address,
                coordinates=location_coords,
                zip_code=zip_code,
                source=source_agency_name,
                original_document_id=message_id,
                original_eido_dict=eido_dict
            )
            logger.info(f"Successfully extracted core data for Report {core_data.report_id[:8]} (Coords: {core_data.coordinates}, ZIP: {core_data.zip_code}, Addr: {core_data.location_address})")
            return core_data
        except Exception as pydantic_error:
             logger.error(f"Failed to create ReportCoreData instance for Msg {message_id}: {pydantic_error}", exc_info=True)
             return None


    def process_report_json(self, json_data: Dict) -> Dict:
        message_id = json_data.get('eidoMessageIdentifier', json_data.get('$id', f"unknown_{str(uuid.uuid4())[:8]}"))
        logger.info(f"--- Processing EIDO Message Dict ID: {message_id} ---")

        if not isinstance(json_data, dict):
            logger.error(f"Input data for message ID '{message_id}' is not a dictionary (Type: {type(json_data)}). Skipping.")
            return {"status": "Input Error: Data must be a JSON object (dictionary).", "message_id": message_id, "incident_id": None, "is_new_incident": False, "summary": None, "actions": None}

        try:
            core_data = self._extract_core_data_from_dict(json_data)
            if not core_data:
                return {"status": "Failed processing: Could not extract core data (check logs for details).", "message_id": message_id, "incident_id": None, "is_new_incident": False, "summary": None, "actions": None}
        except Exception as e:
            logger.error(f"Msg '{message_id}': Unexpected error during core data extraction: {e}", exc_info=True)
            return {"status": f"Processing Error: Core data extraction failed ({type(e).__name__})", "message_id": message_id, "incident_id": None, "is_new_incident": False, "summary": None, "actions": None}

        description_embedding = None 
        if core_data.description and "No description" not in core_data.description:
            try: description_embedding = generate_embedding(core_data.description)
            except Exception as e: logger.warning(f"Msg '{message_id}': Failed to generate embedding: {e}")

        matched_incident = None; matched_incident_id = None; match_score = 0.0; match_reason = "Matching not attempted or failed"
        try:
            active_incidents = incident_store.get_active_incidents()
            logger.debug(f"Msg '{message_id}': Found {len(active_incidents)} active incidents for matching.")
            matched_incident_id, match_score, match_reason = find_match_for_report(core_data, active_incidents)
            if matched_incident_id:
                matched_incident = incident_store.get_incident(matched_incident_id)
                if not matched_incident:
                     logger.error(f"Msg '{message_id}': Matching returned ID {matched_incident_id[:8]} but incident not found! Treating as new.")
                     matched_incident_id = None
        except Exception as e:
             logger.error(f"Msg '{message_id}': Error during incident matching phase: {e}", exc_info=True)
             matched_incident = None; matched_incident_id = None; match_reason = f"Matching Error: {type(e).__name__}"

        incident_to_process = None; is_new_incident = False; incident_id = None
        if matched_incident:
            incident_id = matched_incident.incident_id
            logger.info(f"Msg '{message_id}': Matched to existing Incident {incident_id[:8]} (Score: {match_score:.2f}, Reason: {match_reason}).")
            try:
                 match_info_str = f"Matched Report {core_data.report_id[:8]} (ExtID: {core_data.external_incident_id or 'N/A'}, Score: {match_score:.2f}, Reason: {match_reason})"
                 matched_incident.add_report_core_data(core_data, match_info=match_info_str)
            except Exception as e: logger.error(f"Failed adding report {core_data.report_id[:8]} to incident {incident_id[:8]}: {e}", exc_info=True)
            incident_to_process = matched_incident
            is_new_incident = False
        else:
            logger.info(f"Msg '{message_id}': No match found or error during matching (Reason: {match_reason}). Creating new Incident.")
            try:
                incident_to_process = Incident(incident_type=core_data.incident_type, status="Active")
                match_info_str = f"Created from Report {core_data.report_id[:8]} (ExtID: {core_data.external_incident_id or 'N/A'}, Reason: {match_reason})"
                incident_to_process.add_report_core_data(core_data, match_info=match_info_str)
                incident_id = incident_to_process.incident_id
                logger.info(f"Created new Incident {incident_id[:8]} from Report {core_data.report_id[:8]}.")
                is_new_incident = True
            except Exception as e:
                 logger.error(f"Msg '{message_id}': Failed to create new Incident object or add initial data: {e}", exc_info=True)
                 return {"status": f"Processing Error: Failed to initialize new incident ({type(e).__name__})", "message_id": message_id, "incident_id": None, "is_new_incident": True, "summary": None, "actions": None}

        if incident_to_process:
            try:
                history = ""
                if not is_new_incident and len(incident_to_process.reports_core_data) > 1:
                    history = incident_to_process.get_full_description_history(exclude_latest=True)
                new_summary = summarize_incident(history, core_data)
                if new_summary: incident_to_process.summary = new_summary; logger.debug(f"Incident {incident_id[:8]}: Generated new summary.")
                else:
                    logger.error(f"Incident {incident_id[:8]}: Failed to generate summary.");
                    if not incident_to_process.summary or "not yet generated" in incident_to_process.summary: incident_to_process.summary = f"LLM Error: Could not generate summary. Last report: {core_data.description}"
                new_recommendations = recommend_actions(incident_to_process.summary, core_data)
                if new_recommendations: incident_to_process.recommended_actions = new_recommendations; logger.debug(f"Incident {incident_id[:8]}: Generated new recommendations.")
                else:
                    logger.error(f"Incident {incident_id[:8]}: Failed to generate recommendations.");
                    if not incident_to_process.recommended_actions: incident_to_process.recommended_actions = ["LLM Error: Could not generate recommendations."]
            except Exception as e:
                logger.error(f"Incident {incident_id[:8]}: Error during LLM interaction: {e}", exc_info=True)
                incident_to_process.summary = f"Error during LLM processing: {type(e).__name__}"; incident_to_process.recommended_actions = [f"Error during LLM processing: {type(e).__name__}"]
        else:
             logger.critical(f"Msg '{message_id}': incident_to_process object was None before LLM step. Critical logic error.")
             return {"status": "Processing Error: Internal state error before LLM.", "message_id": message_id, "incident_id": incident_id, "is_new_incident": is_new_incident, "summary": None, "actions": None}

        try:
            incident_store.save_incident(incident_to_process)
            logger.info(f"Msg '{message_id}': Successfully processed. Report {core_data.report_id[:8]} added/updated Incident {incident_id[:8]}.")
            return {"status": "Success", "message_id": message_id, "incident_id": incident_id, "is_new_incident": is_new_incident, "summary": incident_to_process.summary, "actions": incident_to_process.recommended_actions}
        except Exception as e:
             logger.critical(f"CRITICAL: Failed to save incident {incident_id[:8]} to store! Error: {e}", exc_info=True)
             return {"status": f"Processing Error: Failed to save incident to store ({type(e).__name__})", "message_id": message_id, "incident_id": incident_id, "is_new_incident": is_new_incident, "summary": incident_to_process.summary, "actions": incident_to_process.recommended_actions}


    def process_alert_text(self, alert_text: str) -> Union[Dict, List[Dict]]:
        logger.info("--- Processing Raw Alert Text Block ---")
        results: List[Dict] = [] # Ensure results is always a list
        if not alert_text or not isinstance(alert_text, str):
             logger.error("Invalid alert text input: must be a non-empty string.")
             # Return list with one error dict for consistency
             return [{"status": "Input Error: Alert text cannot be empty.", "message_id": "N/A", "incident_id": None}]

        cleaned_text = alert_text.strip()
        if (cleaned_text.startswith('{') and cleaned_text.endswith('}')) or \
           (cleaned_text.startswith('[') and cleaned_text.endswith(']')):
            logger.warning("Input to 'Raw Alert Text' processing looks like JSON. Attempting to process, but for JSON input, use the 'EIDO JSON' tab/endpoint.")

        event_texts: Optional[List[str]] = None
        try:
            event_texts = split_raw_text_into_events(alert_text)
            if event_texts is None: logger.warning("LLM splitting function returned None.")
            elif len(event_texts) == 1: logger.info("LLM splitting function identified a single event.")
            else: logger.info(f"LLM splitting function identified {len(event_texts)} potential events.")
        except Exception as e:
             logger.error(f"Critical error during event splitting call: {e}", exc_info=True)
             event_texts = None

        if not event_texts: 
            logger.warning("Splitting failed or yielded no events. Processing entire block as one event.")
            event_texts = [alert_text]
        
        logger.info(f"Attempting to process {len(event_texts)} event text(s).")

        for i, single_event_text in enumerate(event_texts):
            if not single_event_text.strip():
                logger.warning(f"Skipping empty event text #{i+1}.")
                results.append({"status": f"Processing Error: Event text #{i+1} was empty.", "message_id": f"text_event_{i+1}_empty", "incident_id": None})
                continue
            
            logger.info(f"Processing event text #{i+1}/{len(event_texts)}...")
            generated_eido_dict = parse_alert_to_eido_dict(single_event_text)
            
            if not generated_eido_dict:
                 logger.error(f"Failed to parse event text #{i+1} into EIDO-like structure.")
                 results.append({"status": f"Processing Error: Failed to parse event text #{i+1}.", "message_id": f"text_event_{i+1}_parse_fail", "incident_id": None, "original_text_snippet": single_event_text[:100] + "..."})
                 continue
            
            try:
                result_dict = self.process_report_json(generated_eido_dict) 
                result_dict['source_event_index'] = i + 1
                result_dict['source_event_total'] = len(event_texts)
                result_dict['original_text_snippet'] = single_event_text[:100] + "..." 
                results.append(result_dict)
            except Exception as e:
                 logger.error(f"Error processing dictionary for event text #{i+1}: {e}", exc_info=True)
                 msg_id = generated_eido_dict.get('eidoMessageIdentifier', f'llm_parsed_{i+1}_err')
                 results.append({"status": f"Processing Error: Failed during generated dict processing for event #{i+1} ({type(e).__name__})", "message_id": msg_id, "incident_id": None, "original_text_snippet": single_event_text[:100] + "..."})
        
        return results # Always return a list of result dicts

eido_agent_instance = EidoAgent()