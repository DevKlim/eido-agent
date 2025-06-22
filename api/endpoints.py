import logging
from fastapi import APIRouter, HTTPException, Body, Depends, status, Response
# Renamed to avoid conflict with pydantic Optional
from typing import List, Dict, Any, Union, Optional as TypingOptional
from pydantic import BaseModel, Field
import urllib.parse  # For URL decoding path parameters

from data_models.schemas import Incident as PydanticIncident
from agent.agent_core import eido_agent_instance
from services.storage import IncidentStore
from config.settings import settings
from agent.llm_interface import fill_eido_template
from services import local_geocoder  # Import local_geocoder
import os
import json
# Import schema loading utility
from utils.schema_parser import load_openapi_schema

logger = logging.getLogger(__name__)
app_logger = logging.getLogger("EidoSentinelAPI")
app_logger.setLevel(settings.log_level.upper())

if not app_logger.hasHandlers() and not logging.getLogger().hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)

# Added "EIDO Tools" tag
router = APIRouter(prefix="/api/v1", tags=["Incidents", "Tools", "Geocoding", "EIDO Tools"])

# Models for request/response payloads


class AlertTextPayload(BaseModel):
    alert_text: str = Field(...,
                            description="The raw text content of the alert or report.")


class StatusUpdatePayload(BaseModel):
    status: str = Field(..., example="Resolved",
                        description="The new status for the incident.")


class EidoTemplateFillPayload(BaseModel):
    template_name: str = Field(..., example="traffic_collision.json",
                               description="Filename of the EIDO template.")
    scenario_description: str = Field(
        ..., description="Text description of the scenario to fill the template.")


class LocalGeocodePayload(BaseModel):
    location_name: str = Field(..., example="Geisel Library",
                               description="The name of the location.")
    latitude: float = Field(..., example=32.8811,
                            description="Latitude of the location.")
    longitude: float = Field(..., example=-117.2376,
                             description="Longitude of the location.")
    source: TypingOptional[str] = Field(
        "manual_ui_input", example="manual_ui_input", description="Source of this geocoding entry.")
    notes: TypingOptional[str] = Field(
        "", example="Main entrance", description="Additional notes for this location.")

# New model for saving templates
class EidoTemplateSavePayload(BaseModel):
    filename: str = Field(..., description="The filename for the new template, must end with .json")
    content: Dict[str, Any] = Field(..., description="The JSON content of the template as a dictionary.")


@router.post("/ingest",
             summary="Ingest a single EIDO report (JSON)",
             response_description="Processing result",
             status_code=status.HTTP_201_CREATED)
async def ingest_eido_report(eido_data: Dict = Body(..., example={
    "$id": "urn:emergency:uid:incidentid:a56e556d871:bcf.state.pa.us",
    "lastUpdateTimeStamp": "2021-04-30T14:43:49.439-04:00", "eidoVersion": "1.0",
    "issuingElementIdentification": "idx.state.pa.us",
    "incidentComponent": {"$id": "inc-123", "lastUpdateTimeStamp": "2021-04-30T14:42:00.0-04:00", "incidentTypeCommonRegistryText": "MVAINJY", "locationReference": {"$ref": "loc-123"}},
    "locationComponent": [{"$id": "loc-123", "lastUpdateTimeStamp": "2021-04-30T14:40:00.0-04:00", "locationTypeDescriptionRegistryText": "CurrentIncident", "locationAddressText": "I-80 EAST, MILE MARKER 105.5"}],
    "notesComponent": [{"$id": "note-123", "notesActionComments": "Vehicle rollover with entrapment."}],
    "agencyComponent": [{"$id": "state.pa.us", "lastUpdateTimeStamp": "2021-04-30T14:40:00.0-04:00", "agencyRoleDescriptionRegistryText": ["CallReceiving"], "agencyType": ["psap"]}]
})):
    msg_id_hint = eido_data.get(
        'eidoMessageIdentifier', eido_data.get('$id', 'N/A'))
    app_logger.info(
        f"API /ingest received EIDO JSON data (ID hint: {msg_id_hint}).")
    try:
        result_dict = await eido_agent_instance.process_report_json(eido_data)
        status_message = result_dict.get('status', 'Processing status unknown')

        if result_dict.get('incident_id') and status_message.lower() == "success":
            return {"message": "EIDO report processed successfully.", **result_dict}
        elif status_message.lower().startswith("input error"):
            app_logger.error(f"API /ingest input error: {status_message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed: {status_message}")
        else:
            app_logger.error(f"API /ingest processing error: {status_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed: {status_message}")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        app_logger.critical(
            f"API /ingest unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Unexpected error: {type(e).__name__}")


@router.post("/ingest_alert",
             summary="Ingest raw alert text",
             response_description="List of processing results")
async def ingest_alert_text_endpoint(payload: AlertTextPayload, response: Response):
    alert_text = payload.alert_text
    app_logger.info(
        f"API /ingest_alert received raw alert text (Length: {len(alert_text)}).")
    if not alert_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Field 'alert_text' cannot be empty.")
    try:
        results_union: Union[Dict, List[Dict]] = await eido_agent_instance.process_alert_text(alert_text)

        results_list: List[Dict] = [results_union] if isinstance(results_union, dict) and results_union else (
            results_union if isinstance(results_union, list) else [])

        if not results_list:
            app_logger.error(
                "API /ingest_alert: Agent returned no results for the alert text.")
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="Agent processing failed. The text could not be understood or parsed into any known event format. Please check the text or server logs.")

        all_successful = True
        some_successful = False
        processed_incident_ids = set()

        for res_dict in results_list:
            if isinstance(res_dict, dict):
                status_message = res_dict.get('status', 'Unknown')
                if status_message.lower() == "success":
                    some_successful = True
                    if res_dict.get('incident_id'):
                        processed_incident_ids.add(res_dict.get('incident_id'))
                else:
                    all_successful = False
            else:
                all_successful = False

        response_data = {
            "message": "Alert text processing attempted.",
            "overall_status": "Success" if all_successful else ("Partial Success" if some_successful else "Failure"),
            "processed_incident_ids": list(processed_incident_ids),
            "details": results_list
        }

        if all_successful:
            response.status_code = status.HTTP_201_CREATED
        elif some_successful:
            response.status_code = status.HTTP_207_MULTI_STATUS
        else:
            first_error_message = "All events failed to process. Check server logs for details."
            if results_list and isinstance(results_list[0], dict):
                first_error_message = results_list[0].get('status', first_error_message)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Input could not be processed. Reason: {first_error_message}",
            )

        return response_data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        app_logger.critical(
            f"API /ingest_alert unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Unexpected error: {type(e).__name__}")

db_incident_store = IncidentStore()


@router.get("/incidents", response_model=List[PydanticIncident], summary="List all incidents")
async def get_all_incidents_endpoint():
    app_logger.info("API request received for /incidents")
    return await db_incident_store.get_all_incidents()


@router.get("/incidents/active", response_model=List[PydanticIncident], summary="List active incidents")
async def get_active_incidents_endpoint():
    app_logger.info("API request received for /incidents/active")
    return await db_incident_store.get_active_incidents()


@router.get("/incidents/{incident_id}", response_model=PydanticIncident, summary="Get incident details", responses={404: {"description": "Incident not found"}})
async def get_incident_details_endpoint(incident_id: str):
    app_logger.info(f"API request received for /incidents/{incident_id}")
    incident = await db_incident_store.get_incident(incident_id)
    if incident:
        return incident
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Incident with ID '{incident_id}' not found.")


@router.put("/incidents/{incident_id}/status", summary="Update incident status", tags=["Incidents"], responses={404: {"description": "Incident not found"}, 400: {"description": "Invalid status"}})
async def update_incident_status_endpoint(incident_id: str, payload: StatusUpdatePayload):
    new_status = payload.status
    app_logger.info(
        f"API request to update status for Incident {incident_id} to '{new_status}'")

    allowed_statuses = ["Active", "Updated",
                        "Monitoring", "Resolved", "Closed"]
    if new_status not in allowed_statuses:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid status value '{new_status}'. Allowed: {allowed_statuses}")

    success = await db_incident_store.update_incident_status(incident_id, new_status)
    if success:
        return {"message": f"Incident {incident_id} status updated to '{new_status}'."}
    else:
        incident_check = await db_incident_store.get_incident(incident_id)
        if not incident_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Incident with ID '{incident_id}' not found for status update.")
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to update incident status.")


@router.delete("/admin/clear_store", summary="Clear all incidents (Admin)", tags=["Admin"], status_code=status.HTTP_200_OK)
async def clear_incident_store_endpoint():
    app_logger.warning(
        "API request received to clear the entire incident store.")
    try:
        await db_incident_store.clear_store()
        return {"message": "Incident store cleared successfully."}
    except Exception as e:
        app_logger.error(
            f"Failed to clear incident store via API: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to clear incident store.")


@router.post("/generate_eido_from_template",
             summary="Generate EIDO JSON from template and scenario",
             response_description="Generated EIDO JSON string or error",
             tags=["EIDO Tools"])
async def generate_eido_from_template_endpoint(payload: EidoTemplateFillPayload):
    app_logger.info(
        f"API /generate_eido_from_template called for template: {payload.template_name}")
    project_root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    template_dir_path = os.path.join(project_root_dir, "eido_templates")
    template_path = os.path.join(template_dir_path, payload.template_name)

    if not os.path.exists(template_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Template '{payload.template_name}' not found.")
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        app_logger.error(
            f"Error reading template file '{payload.template_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Error reading template file.")

    generated_json_str = await fill_eido_template(
        template_content, payload.scenario_description)
        
    if generated_json_str:
        try:
            parsed_json = json.loads(generated_json_str)
            return {"generated_eido": parsed_json}
        except json.JSONDecodeError:
            app_logger.error(
                f"LLM generated non-JSON output for template '{payload.template_name}'. Output: {generated_json_str[:200]}...")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="LLM generated invalid JSON output.")
    else:
        app_logger.error(
            f"LLM failed to fill template '{payload.template_name}'.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="LLM failed to generate EIDO from template.")


@router.post("/tools/geocoding/local_store", summary="Add or update a location in the local geocoder store", tags=["Tools", "Geocoding"])
async def update_local_geocode_entry(payload: LocalGeocodePayload):
    app_logger.info(
        f"API request to update local geocode store for: {payload.location_name}")
    success = local_geocoder.update_known_location(
        payload.location_name,
        payload.latitude,
        payload.longitude,
        source=payload.source if payload.source is not None else "manual_ui_input",
        notes=payload.notes if payload.notes is not None else ""
    )
    if success:
        return {"message": f"Location '{payload.location_name}' updated/added to local store."}
    else:
        # local_geocoder.update_known_location logs errors internally
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Failed to update local geocode entry. Check server logs for details.")


@router.get("/tools/geocoding/local_store", summary="List all known locations in the local geocoder store", response_model=Dict[str, Any], tags=["Tools", "Geocoding"])
async def list_local_geocode_entries():
    app_logger.info("API request to list all local geocode entries.")
    return local_geocoder.get_all_known_locations()


@router.delete("/tools/geocoding/local_store/{location_name}", summary="Remove a location from the local geocoder store", tags=["Tools", "Geocoding"], status_code=status.HTTP_204_NO_CONTENT)
async def delete_local_geocode_entry(location_name: str):
    try:
        decoded_location_name = urllib.parse.unquote(location_name)
    except Exception as e:
        app_logger.warning(
            f"Failed to URL decode location_name '{location_name}': {e}. Using as is.")
        decoded_location_name = location_name

    app_logger.info(
        f"API request to delete local geocode entry: {decoded_location_name}")
    success = local_geocoder.remove_known_location(decoded_location_name)
    if success:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Location '{decoded_location_name}' not found in local store or removal failed.")

# --- New Endpoints for EIDO Template Editor ---

@router.get("/tools/eido/schema", summary="Get full EIDO component schema", tags=["EIDO Tools"])
async def get_eido_schema():
    app_logger.info("API request for EIDO component schema.")
    schema = load_openapi_schema()
    if not schema:
        raise HTTPException(status_code=500, detail="EIDO schema file not found or invalid on server.")
    
    components = schema.get('components', {}).get('schemas', {})
    if not components:
        raise HTTPException(status_code=404, detail="No components found in schema.")
        
    return components

@router.post("/tools/eido/templates", summary="Save a new EIDO template", tags=["EIDO Tools"], status_code=status.HTTP_201_CREATED)
async def save_eido_template(payload: EidoTemplateSavePayload):
    app_logger.info(f"API request to save new EIDO template: {payload.filename}")
    
    # Security validation
    if not payload.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Filename must end with .json")
    
    clean_filename = os.path.basename(payload.filename)
    if clean_filename != payload.filename:
        raise HTTPException(status_code=400, detail="Invalid filename. It cannot contain path characters.")

    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    template_dir = os.path.join(project_root_dir, "eido_templates")
    os.makedirs(template_dir, exist_ok=True)
    
    file_path = os.path.join(template_dir, clean_filename)

    if os.path.exists(file_path):
        raise HTTPException(status_code=409, detail=f"Template file '{clean_filename}' already exists.")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload.content, f, indent=4)
        return {"message": f"Template '{clean_filename}' saved successfully."}
    except Exception as e:
        app_logger.error(f"Failed to save template file '{clean_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to write template file to server.")