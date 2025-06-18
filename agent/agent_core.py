import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any, Union
import uuid
import re
from pydantic import ValidationError

from data_models.schemas import ReportCoreData, Incident as PydanticIncident
from services.storage import IncidentStore
from services import local_geocoder
from agent.matching import find_match_for_report
from agent.llm_interface import (
    summarize_incident, recommend_actions, split_raw_text_into_events,
    extract_eido_from_alert_text, generate_incident_name
)
from services.advanced_geocoding_service import get_advanced_geocoding_service, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_NONE
from utils.helpers import parse_civic_address_from_pidf, format_address_from_components
from agent.alert_parser import parse_alert_to_eido_dict

logger = logging.getLogger(__name__)


def _ensure_timezone_aware(ts_input: Any) -> datetime:
    if isinstance(ts_input, datetime):
        return ts_input if ts_input.tzinfo else ts_input.replace(tzinfo=timezone.utc)
    if isinstance(ts_input, str):
        try:
            dt = datetime.fromisoformat(ts_input.replace('Z', '+00:00'))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)


class EidoAgent:
    def __init__(self):
        logger.info("EIDO Agent Core initialized.")
        self.incident_store = IncidentStore()

    def _resolve_ref_string_from_dict(self, ref_input: Any) -> Optional[str]:
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
                    logger.warning(
                        f"Reference field contained multiple items, using first $ref: '{ref_id}'")
            elif isinstance(first_item, str) and first_item.startswith("$ref:"):
                ref_id = first_item.split(':', 1)[1]
                if len(ref_input) > 1:
                    logger.warning(
                        f"Reference field contained multiple strings, using first $ref: '{ref_id}'")
        return ref_id

    async def _attempt_geocode_and_update_store(self, text_to_geocode: str) -> Optional[Tuple[float, float]]:
        if not text_to_geocode or not isinstance(text_to_geocode, str) or not text_to_geocode.strip():
            logger.debug(
                f"Advanced Geocoding: Skipped due to empty or invalid text: '{text_to_geocode}'")
            return None
        logger.info(
            f"Advanced Geocoding: Attempting for text: '{text_to_geocode[:100]}...'")
        adv_geo_service = await get_advanced_geocoding_service()
        geocoding_result = await adv_geo_service.geocode_narrative(text_to_geocode)
        coords: Optional[Tuple[float, float]
                         ] = geocoding_result.get("coordinates")
        confidence: str = geocoding_result.get("confidence", CONFIDENCE_NONE)
        method: str = geocoding_result.get("method", "Unknown")
        for log_entry in geocoding_result.get("reasoning_log", []):
            logger.debug(f"AdvGeoLog: {log_entry}")
        if coords:
            logger.info(
                f"Advanced Geocoding successful for '{text_to_geocode[:100]}...': {coords}. Confidence: {confidence}, Method: {method}")
            if confidence in [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM]:
                try:
                    local_geocoder.update_known_location(
                        text_to_geocode, coords[0], coords[1],
                        source=f"adv_geo_{method.replace(' ', '_').lower()}",
                        notes=f"Confidence: {confidence}. Original: '{text_to_geocode[:100]}...'"
                    )
                    logger.info(
                        f"Cached advanced geocoding result for '{text_to_geocode[:100]}...' in local store.")
                except Exception as e:
                    logger.error(
                        f"Failed to cache advanced geocoding result for '{text_to_geocode[:100]}...': {e}", exc_info=True)
            else:
                logger.info(
                    f"Advanced geocoding result for '{text_to_geocode[:100]}...' not cached due to confidence: {confidence}.")
            return coords
        else:
            logger.warning(
                f"Advanced Geocoding failed for '{text_to_geocode[:100]}...'. Method: {method}, Confidence: {confidence}.")
            return None

    def _extract_core_data_from_dict(self, eido_dict: Dict) -> Optional[ReportCoreData]:
        if not isinstance(eido_dict, dict):
            logger.error(
                f"Input to _extract_core_data_from_dict is not a dictionary (Type: {type(eido_dict)}).")
            return None

        message_id = eido_dict.get('eidoMessageIdentifier', eido_dict.get(
            '$id', f"unknown_{str(uuid.uuid4())[:8]}"))
        logger.info(
            f"Extracting core data from EIDO Message Dict: {message_id}")

        primary_comp, source_comp_type = None, "Unknown"
        for comp_type_key in ['incidentComponent', 'callComponent']:
            comp_list = eido_dict.get(comp_type_key)
            if isinstance(comp_list, list) and comp_list:
                primary_comp = next(
                    (comp for comp in comp_list if isinstance(comp, dict)), None)
                if primary_comp:
                    source_comp_type = comp_type_key.replace(
                        'Component', 'Component').capitalize()
                    break

        if not primary_comp:
            logger.error(
                f"Msg {message_id}: CRITICAL FAILURE - No suitable primary component found in EIDO dict.")
            return None

        incident_tracking_id = primary_comp.get(
            'incidentTrackingIdentifier') or primary_comp.get('callTrackingIdentifier')
        ts_value = primary_comp.get(
            'lastUpdateTimeStamp', eido_dict.get('lastUpdateTimeStamp'))
        timestamp = _ensure_timezone_aware(
            ts_value or datetime.now(timezone.utc))

        # --- IMPROVED: More robust incident type extraction ---
        raw_incident_type = (
            primary_comp.get('incidentTypeCommonRegistryText') or
            primary_comp.get('callTypeCommonRegistryText') or
            # Added this field from sample
            primary_comp.get('standardPrimaryCallType')
        )
        incident_type = "Unknown"  # Default value
        if isinstance(raw_incident_type, list) and raw_incident_type:
            incident_type = str(raw_incident_type[0])
        elif isinstance(raw_incident_type, str) and raw_incident_type.strip():
            incident_type = raw_incident_type.strip()

        # --- IMPROVED: Gracefully handle missing descriptions ---
        descriptions = []
        for comp_key in ['notesComponent', 'commentsComponent']:
            for item in eido_dict.get(comp_key, []):
                if isinstance(item, dict):
                    text = item.get('noteText') or item.get('commentText')
                    if text:
                        # Ensure text is a string
                        descriptions.append(str(text))

        # Add incident summary text if available
        summary_text = primary_comp.get('incidentSummaryText')
        if summary_text:
            descriptions.append(str(summary_text))

        # Use None if no description is found
        full_description = "\n".join(descriptions) if descriptions else None

        location_address, location_coords, zip_code, location_narrative_for_geocoding = None, None, None, None
        loc_ref_id = self._resolve_ref_string_from_dict(
            primary_comp.get('locationReference'))
        location_components = eido_dict.get('locationComponent', [])
        primary_loc_comp = next((lc for lc in location_components if isinstance(
            lc, dict) and lc.get('$id') == loc_ref_id), None)
        if not primary_loc_comp and location_components:
            primary_loc_comp = next(
                (lc for lc in location_components if isinstance(lc, dict)), None)

        if primary_loc_comp:
            loc_val = primary_loc_comp.get('locationByValue')
            if isinstance(loc_val, str) and loc_val.strip().startswith('<?xml'):
                addr_components = parse_civic_address_from_pidf(loc_val)
                location_address = format_address_from_components(
                    addr_components) if addr_components else None
            else:
                location_address = primary_loc_comp.get('locationAddressText') or (
                    loc_val if isinstance(loc_val, str) else None)

            location_narrative_for_geocoding = primary_loc_comp.get(
                'locationNotes') or location_address
            if location_narrative_for_geocoding:
                coord_match = re.search(
                    r'(-?\d{1,2}\.\d{3,})\s*[, ]\s*(-?\d{1,3}\.\d{3,})', location_narrative_for_geocoding)
                if coord_match:
                    location_coords = (
                        float(coord_match.group(1)), float(coord_match.group(2)))
                zip_match = re.search(
                    r'\b(\d{5}(?:-\d{4})?)\b', location_narrative_for_geocoding)
                if zip_match:
                    zip_code = zip_match.group(1)

        source_agency_name = eido_dict.get(
            'sendingSystemIdentifier', "Unknown Source")

        try:
            core_data = ReportCoreData(
                external_incident_id=incident_tracking_id, timestamp=timestamp, incident_type=incident_type,
                description=full_description, location_address=location_address, coordinates=location_coords,
                zip_code=zip_code, source=source_agency_name, original_document_id=message_id,
                original_eido_dict=eido_dict
            )
            # FIX: Assign extra fields directly to the instance, not to model_extra
            # This works because ConfigDict has extra='allow' in schemas.py
            if location_narrative_for_geocoding:
                core_data.location_narrative_for_geocoding = location_narrative_for_geocoding  # type: ignore
            return core_data
        except ValidationError as p_err:
            logger.error(
                f"Pydantic validation error creating ReportCoreData for Msg {message_id}: {p_err}", exc_info=True)
            return None

    async def _process_core_data(self, core_data: ReportCoreData) -> Dict:
        """Central processing logic for a validated ReportCoreData object."""
        message_id = core_data.original_document_id
        logger.info(
            f"Processing Core Data for Report ID: {core_data.report_id[:8]} (Orig. Msg ID: {message_id})")

        if not core_data.coordinates:
            logger.info(
                f"Msg {message_id}: Coordinates not found. Attempting advanced geocoding.")
            # FIX: Safely get the extra attribute
            narrative = getattr(
                core_data, 'location_narrative_for_geocoding', None)
            text_for_geocoding = narrative or core_data.description
            if text_for_geocoding:
                geocoded_coords = await self._attempt_geocode_and_update_store(str(text_for_geocoding))
                if geocoded_coords:
                    core_data.coordinates = geocoded_coords
                    logger.info(
                        f"Msg {message_id}: Advanced geocoding successful. Updated Coords: {core_data.coordinates}")

        try:
            active_incidents = await self.incident_store.get_active_incidents()
            matched_id, score, reason = find_match_for_report(
                core_data, active_incidents)
        except Exception as e:
            logger.error(
                f"Msg '{message_id}': Error during incident matching: {e}", exc_info=True)
            matched_id, score, reason = None, 0.0, f"Matching Error: {type(e).__name__}"

        is_new = False
        if matched_id and (matched_incident := await self.incident_store.get_incident(matched_id)):
            incident_to_process = matched_incident
            logger.info(
                f"Msg '{message_id}': Matched to existing Incident {matched_id[:8]} (Score: {score:.2f}).")
            match_info = f"Matched Report {core_data.report_id[:8]} (ExtID: {core_data.external_incident_id or 'N/A'}, Score: {score:.2f}, Reason: {reason})"
        else:
            incident_to_process = PydanticIncident(
                incident_type=core_data.incident_type, status="Active")
            is_new = True
            logger.info(
                f"Msg '{message_id}': No match found. Creating new Incident {incident_to_process.incident_id[:8]}.")
            match_info = f"Created from Report {core_data.report_id[:8]} (ExtID: {core_data.external_incident_id or 'N/A'}, Reason: {reason})"

        incident_to_process.add_report_core_data(
            core_data, match_info=match_info)

        try:
            history = incident_to_process.get_full_description_history(
                exclude_latest=True) if not is_new else ""
            incident_to_process.summary = summarize_incident(
                history, core_data) or incident_to_process.summary
            incident_to_process.recommended_actions = recommend_actions(
                incident_to_process.summary, core_data) or incident_to_process.recommended_actions
            
            # Generate a name if it's new or doesn't have one
            if not incident_to_process.name or incident_to_process.name == "Untitled Incident":
                location_context = core_data.location_address or (incident_to_process.addresses[0] if incident_to_process.addresses else "Unknown Location")
                incident_to_process.name = generate_incident_name(
                    incident_to_process.incident_type or "Incident",
                    location_context,
                    incident_to_process.summary
                ) or incident_to_process.name
                logger.info(f"Generated name for Incident {incident_to_process.incident_id[:8]}: '{incident_to_process.name}'")

        except Exception as e:
            logger.error(
                f"Incident {incident_to_process.incident_id[:8]}: Error during LLM interaction: {e}", exc_info=True)

        await self.incident_store.save_incident(incident_to_process)
        logger.info(
            f"Msg '{message_id}': Successfully processed. Report {core_data.report_id[:8]} -> Incident {incident_to_process.incident_id[:8]}.")
        return {"status": "Success", "message_id": message_id, "incident_id": incident_to_process.incident_id, "is_new_incident": is_new, "summary": incident_to_process.summary, "actions": incident_to_process.recommended_actions}

    async def process_report_json(self, json_data: Dict) -> Dict:
        """Processes a full EIDO JSON dictionary by extracting core data first."""
        message_id = json_data.get(
            'eidoMessageIdentifier', json_data.get('$id', 'unknown'))
        logger.info(f"--- Processing EIDO Message Dict ID: {message_id} ---")
        core_data = self._extract_core_data_from_dict(json_data)
        if not core_data:
            return {"status": "Failed processing: Could not extract core data.", "message_id": message_id, "incident_id": None}

        return await self._process_core_data(core_data)

    async def process_alert_text(self, alert_text: str) -> Union[Dict, List[Dict]]:
        """Processes raw alert text by extracting structured data and creating ReportCoreData directly."""
        logger.info(
            "--- Processing Raw Alert Text Block (New, Robust Pipeline) ---")
        if not alert_text or not isinstance(alert_text, str):
            return [{"status": "Input Error: Alert text cannot be empty."}]

        event_texts = split_raw_text_into_events(alert_text) or [alert_text]
        logger.info(f"Attempting to process {len(event_texts)} event text(s).")
        results = []

        for i, single_event_text in enumerate(event_texts):
            if not single_event_text.strip():
                continue

            # This now calls the alert_parser, which creates a full EIDO-like dictionary.
            # This is a more robust pattern, centralizing the parsing logic.
            eido_dict = parse_alert_to_eido_dict(single_event_text)

            if not eido_dict:
                logger.error(
                    f"Event {i+1}: Failed to parse alert text into an EIDO-like dictionary.")
                results.append(
                    {"status": "Failed processing: Could not parse text with LLM.", "source_event_index": i + 1})
                continue

            message_id = eido_dict.get('eidoMessageIdentifier', eido_dict.get(
                '$id', f"llm_parsed_{str(uuid.uuid4())[:8]}"))
            try:
                # The alert_parser has already created a compatible dictionary.
                # Now we process it just like a regular EIDO JSON.
                core_data = self._extract_core_data_from_dict(eido_dict)
                if not core_data:
                    logger.error(
                        f"Event {i+1} (Msg: {message_id}): Could not extract core data from LLM-generated dict.")
                    results.append({"status": "Failed processing: Core data extraction failed from generated dict.",
                                   "message_id": message_id, "incident_id": None, "source_event_index": i + 1})
                    continue

                result_dict = await self._process_core_data(core_data)
                result_dict['source_event_index'] = i + 1
                results.append(result_dict)

            except Exception as e:
                logger.error(
                    f"Event {i+1} (Msg: {message_id}): An unexpected error occurred during raw text processing. Error: {e}", exc_info=True)
                results.append({"status": f"Failed processing: Unexpected error ({type(e).__name__}).",
                               "message_id": message_id, "incident_id": None, "source_event_index": i + 1})

        return results if len(results) > 1 else (results[0] if results else {})


eido_agent_instance = EidoAgent()