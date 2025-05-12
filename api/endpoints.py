import logging
from fastapi import APIRouter, HTTPException, Body, Depends, status, Response
from typing import List, Dict, Any, Union 
from pydantic import BaseModel, Field

from data_models.schemas import Incident as PydanticIncident, ReportCoreData
from agent.agent_core import eido_agent_instance
from services.storage import IncidentStore # Using the class, not instance
from config.settings import settings
from agent.llm_interface import fill_eido_template # For the new generator endpoint
import os
import json

logger = logging.getLogger(__name__)
app_logger = logging.getLogger("EidoSentinelAPI") 
app_logger.setLevel(settings.log_level.upper())

# Ensure handler is added if not already (e.g. by uvicorn)
if not app_logger.hasHandlers() and not logging.getLogger().hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)


router = APIRouter(prefix="/api/v1", tags=["Incidents"])

# Models for request/response payloads
class AlertTextPayload(BaseModel):
    alert_text: str = Field(..., description="The raw text content of the alert or report.")

class StatusUpdatePayload(BaseModel):
    status: str = Field(..., example="Resolved", description="The new status for the incident.")

class EidoTemplateFillPayload(BaseModel):
    template_name: str = Field(..., example="traffic_collision.json", description="Filename of the EIDO template.")
    scenario_description: str = Field(..., description="Text description of the scenario to fill the template.")

# Dependency to get an instance of IncidentStore (which now uses DB)
# We don't need to pass the store itself if methods are static or agent handles it.
# Agent now has its own store instance.
# We can use the global `eido_agent_instance` for processing.

@router.post("/ingest",
             summary="Ingest a single EIDO report (JSON)",
             response_description="Processing result",
             status_code=status.HTTP_201_CREATED)
async def ingest_eido_report(eido_data: Dict = Body(..., example={
                                                        "eidoMessageIdentifier": "msg_example_123", "$id": "msg_example_123",
                                                        "sendingSystemIdentifier": "CADSystemX", "lastUpdateTimeStamp": "2024-10-26T10:00:00Z",
                                                        "incidentComponent": [{"componentIdentifier": "inc_123", "incidentTrackingIdentifier": "FIRE2024-001", "lastUpdateTimeStamp": "2024-10-26T10:00:00Z", "incidentTypeCommonRegistryText": "Structure Fire", "locationReference": "$ref:loc_123"}],
                                                        "locationComponent": [{"$id": "loc_123", "componentIdentifier": "loc_123", "locationByValue": "<?xml version='1.0' encoding='UTF-8'?><location><civicAddressText>123 University Ave, Springfield, IL 62704</civicAddressText><gml:Point xmlns:gml='http://www.opengis.net/gml'><gml:pos>39.8010 -89.6437</gml:pos></gml:Point></location>"}],
                                                        "notesComponent": [{"componentIdentifier": "note_123", "noteDateTimeStamp": "2024-10-26T10:00:05Z", "noteText": "Caller reports smoke..."}]
                                                        })):
    msg_id_hint = eido_data.get('eidoMessageIdentifier', eido_data.get('$id', 'N/A'))
    app_logger.info(f"API /ingest received EIDO JSON data (ID hint: {msg_id_hint}).")
    try:
        # eido_agent_instance methods are now async
        result_dict = await eido_agent_instance.process_report_json(eido_data)
        status_message = result_dict.get('status', 'Processing status unknown')

        if result_dict.get('incident_id') and status_message.lower() == "success":
            return {"message": "EIDO report processed successfully.", **result_dict}
        elif status_message.lower().startswith("input error"):
             app_logger.error(f"API /ingest input error: {status_message}")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed: {status_message}")
        else:
            app_logger.error(f"API /ingest processing error: {status_message}")
            # Use a more specific error code if agent indicates specific failure types
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed: {status_message}")
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        app_logger.critical(f"API /ingest unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {type(e).__name__}")


@router.post("/ingest_alert",
             summary="Ingest raw alert text",
             response_description="List of processing results")
async def ingest_alert_text_endpoint(payload: AlertTextPayload, response: Response): # Renamed to avoid conflict
    alert_text = payload.alert_text
    app_logger.info(f"API /ingest_alert received raw alert text (Length: {len(alert_text)}).")
    if not alert_text:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Field 'alert_text' cannot be empty.")
    try:
        # eido_agent_instance methods are now async
        results_union: Union[Dict, List[Dict]] = await eido_agent_instance.process_alert_text(alert_text)
        
        results_list: List[Dict] = [results_union] if isinstance(results_union, dict) else (results_union if isinstance(results_union, list) else [])

        if not results_list: 
            app_logger.error("API /ingest_alert: Agent returned no results for the alert text.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Agent processing yielded no results.")

        all_successful = True; some_successful = False
        error_messages = []
        processed_incident_ids = set()

        for res_dict in results_list:
            if isinstance(res_dict, dict):
                status_message = res_dict.get('status', 'Unknown')
                if status_message.lower() == "success":
                    some_successful = True
                    if res_dict.get('incident_id'): processed_incident_ids.add(res_dict.get('incident_id'))
                else:
                    all_successful = False; error_messages.append(res_dict.get('status_detail', status_message))
            else: all_successful = False; error_messages.append("Invalid result format for an event.")
        
        response_data = {
            "message": "Alert text processing attempted.",
            "overall_status": "Success" if all_successful else ("Partial Success" if some_successful else "Failure"),
            "processed_incident_ids": list(processed_incident_ids),
            "details": results_list 
        }
        
        response.status_code = status.HTTP_201_CREATED if all_successful else (status.HTTP_207_MULTI_STATUS if some_successful else status.HTTP_500_INTERNAL_SERVER_ERROR)
        return response_data

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        app_logger.critical(f"API /ingest_alert unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {type(e).__name__}")

# New store instance for direct DB access from endpoints
db_incident_store = IncidentStore()

@router.get("/incidents", response_model=List[PydanticIncident], summary="List all incidents")
async def get_all_incidents_endpoint(): # Renamed
    app_logger.info("API request received for /incidents")
    return await db_incident_store.get_all_incidents()

@router.get("/incidents/active", response_model=List[PydanticIncident], summary="List active incidents")
async def get_active_incidents_endpoint(): # Renamed
    app_logger.info("API request received for /incidents/active")
    return await db_incident_store.get_active_incidents()

@router.get("/incidents/{incident_id}", response_model=PydanticIncident, summary="Get incident details", responses={404: {"description": "Incident not found"}})
async def get_incident_details_endpoint(incident_id: str): # Renamed
    app_logger.info(f"API request received for /incidents/{incident_id}")
    incident = await db_incident_store.get_incident(incident_id)
    if incident: return incident
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")

@router.put("/incidents/{incident_id}/status", summary="Update incident status", tags=["Admin"], responses={404: {"description": "Incident not found"}, 400: {"description": "Invalid status"}})
async def update_incident_status_endpoint(incident_id: str, payload: StatusUpdatePayload): # Renamed
    new_status = payload.status
    app_logger.info(f"API request to update status for Incident {incident_id} to '{new_status}'")
    
    allowed_statuses = ["Active", "Updated", "Monitoring", "Resolved", "Closed"] 
    if new_status not in allowed_statuses:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status value '{new_status}'. Allowed: {allowed_statuses}")
    
    success = await db_incident_store.update_incident_status(incident_id, new_status)
    if success:
        return {"message": f"Incident {incident_id} status updated to '{new_status}'."}
    else:
        # update_incident_status logs warnings for non-existent incident or other failures.
        # If it returns False, it means either not found or another issue.
        # Check if incident exists first to give specific 404.
        incident_check = await db_incident_store.get_incident(incident_id)
        if not incident_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found for status update.")
        else: # Other failure during update
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update incident status.")


@router.delete("/admin/clear_store", summary="Clear all incidents (Admin)", tags=["Admin"], status_code=status.HTTP_200_OK)
async def clear_incident_store_endpoint(): # Renamed
    app_logger.warning("API request received to clear the entire incident store.")
    try:
        await db_incident_store.clear_store()
        return {"message": f"Incident store cleared successfully."}
    except Exception as e:
         app_logger.error(f"Failed to clear incident store via API: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear incident store.")

# Endpoint for EIDO Generator
@router.post("/generate_eido_from_template",
             summary="Generate EIDO JSON from template and scenario",
             response_description="Generated EIDO JSON string or error",
             tags=["EIDO Tools"])
async def generate_eido_from_template_endpoint(payload: EidoTemplateFillPayload):
    app_logger.info(f"API /generate_eido_from_template called for template: {payload.template_name}")
    
    # Construct path to template
    # Ensure TEMPLATE_DIR is accessible here or pass as config
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    template_dir_path = os.path.join(project_root_dir, "eido_templates")
    template_path = os.path.join(template_dir_path, payload.template_name)

    if not os.path.exists(template_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template '{payload.template_name}' not found.")

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        app_logger.error(f"Error reading template file '{payload.template_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error reading template file.")

    # LLM call through llm_interface
    generated_json_str = fill_eido_template(template_content, payload.scenario_description) # This is synchronous

    if generated_json_str:
        try:
            # Validate if it's JSON before returning
            parsed_json = json.loads(generated_json_str)
            return {"generated_eido": parsed_json} # Return parsed JSON
        except json.JSONDecodeError:
            app_logger.error(f"LLM generated non-JSON output for template '{payload.template_name}'. Output: {generated_json_str[:200]}...")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM generated invalid JSON output.")
    else:
        app_logger.error(f"LLM failed to fill template '{payload.template_name}'.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM failed to generate EIDO from template.")