import logging
from fastapi import APIRouter, HTTPException, Body, Depends, status, Response
from typing import List, Dict, Any, Union 
from pydantic import BaseModel, Field

from data_models.schemas import Incident
from agent.agent_core import eido_agent_instance
from services.storage import incident_store
from config.settings import settings

logger = logging.getLogger(__name__)
# BasicConfig might conflict with uvicorn's. FastAPI/Uvicorn manage their logging.
# We ensure our app's logger respects settings.log_level.
app_logger = logging.getLogger("EidoSentinelAPI") # Specific logger for API parts
app_logger.setLevel(settings.log_level.upper())
# Add a handler if not already configured at a higher level or by uvicorn
if not app_logger.hasHandlers() and not logging.getLogger().hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)
    # Also configure root to catch uvicorn/fastapi low-level logs if needed
    # logging.getLogger().addHandler(handler) 
    # logging.getLogger().setLevel(settings.log_level.upper())


router = APIRouter(prefix="/api/v1", tags=["Incidents"])

class AlertTextPayload(BaseModel):
    alert_text: str = Field(..., description="The raw text content of the alert or report.")

# --- API Endpoints ---

@router.post("/ingest",
             summary="Ingest a single EIDO report (JSON)",
             description="Receives an EIDO report in JSON dictionary format, processes it using the agent, and returns the result.",
             status_code=status.HTTP_201_CREATED,
             response_description="Processing result including incident ID and status message")
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
        result_dict = eido_agent_instance.process_report_json(eido_data)
        incident_id = result_dict.get('incident_id')
        status_message = result_dict.get('status', 'Processing status unknown')

        if incident_id and status_message.lower() == "success":
            response_data = {"message": "EIDO report processed successfully.", **result_dict}
            return response_data
        elif status_message.lower().startswith("input error"):
             app_logger.error(f"API /ingest input error: {status_message}")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed: {status_message}")
        else:
            app_logger.error(f"API /ingest processing error: {status_message}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed: {status_message}")
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        app_logger.critical(f"API /ingest unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {type(e).__name__}")


@router.post("/ingest_alert",
             summary="Ingest raw alert text (potentially multiple events)",
             description="Receives raw alert text. The agent attempts to split it into individual events, parse each into an EIDO-like structure, process them, and return results.",
             response_description="List of processing results for each identified event.")
async def ingest_alert_text(payload: AlertTextPayload, response: Response):
    alert_text = payload.alert_text
    app_logger.info(f"API /ingest_alert received raw alert text (Length: {len(alert_text)}).")
    if not alert_text:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Field 'alert_text' cannot be empty.")

    try:
        results_union: Union[Dict, List[Dict]] = eido_agent_instance.process_alert_text(alert_text)
        
        # Standardize response to always be a list for this endpoint
        results_list: List[Dict] = []
        if isinstance(results_union, dict): 
            results_list = [results_union]
        elif isinstance(results_union, list):
            results_list = results_union
        else: # Should not happen if agent returns Dict or List[Dict]
            app_logger.error(f"API /ingest_alert: Agent returned unexpected type: {type(results_union)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Agent processing yielded an invalid result format.")


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
                    all_successful = False
                    error_messages.append(res_dict.get('status_detail', status_message))
            else: 
                all_successful = False; error_messages.append("Invalid result format for an event from agent.")
        
        response_data = {
            "message": "Alert text processing attempted.",
            "overall_status": "Success" if all_successful else ("Partial Success" if some_successful else "Failure"),
            "processed_incident_ids": list(processed_incident_ids),
            "details": results_list 
        }
        
        # Determine appropriate status code for the HTTP response object
        if all_successful:
            response.status_code = status.HTTP_201_CREATED
        elif some_successful:
            response.status_code = status.HTTP_207_MULTI_STATUS
        else: # Complete failure
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            # Update detail for HTTPException if we were to raise it
            # detail_msg = "; ".join(error_messages) if error_messages else "Processing failed for all events."
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_msg)

        return response_data

    except HTTPException as http_exc: raise http_exc # Re-raise if already an HTTPException
    except Exception as e:
        app_logger.critical(f"API /ingest_alert unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {type(e).__name__}")


@router.get("/incidents", response_model=List[Incident], summary="List all incidents")
async def get_all_incidents():
    app_logger.info("API request received for /incidents")
    return incident_store.get_all_incidents()

@router.get("/incidents/active", response_model=List[Incident], summary="List active incidents")
async def get_active_incidents():
    app_logger.info("API request received for /incidents/active")
    return incident_store.get_active_incidents()

@router.get("/incidents/{incident_id}", response_model=Incident, summary="Get incident details", responses={404: {"description": "Incident not found"}})
async def get_incident_details(incident_id: str):
    app_logger.info(f"API request received for /incidents/{incident_id}")
    incident = incident_store.get_incident(incident_id)
    if incident: return incident
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")

@router.get("/incidents/{incident_id}/summary", summary="Get incident summary", responses={404: {"description": "Incident not found"}})
async def get_incident_summary_api(incident_id: str):
    app_logger.info(f"API request received for /incidents/{incident_id}/summary")
    incident = incident_store.get_incident(incident_id)
    if incident: return {"incident_id": incident_id, "summary": incident.summary}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")

@router.get("/incidents/{incident_id}/recommendations", summary="Get recommended actions", responses={404: {"description": "Incident not found"}})
async def get_incident_recommendations_api(incident_id: str):
    app_logger.info(f"API request received for /incidents/{incident_id}/recommendations")
    incident = incident_store.get_incident(incident_id)
    if incident: return {"incident_id": incident_id, "recommended_actions": incident.recommended_actions}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")

class StatusUpdatePayload(BaseModel):
    status: str = Field(..., example="Resolved", description="The new status for the incident.")

@router.put("/incidents/{incident_id}/status", summary="Update incident status", tags=["Admin"], responses={404: {"description": "Incident not found"}, 400: {"description": "Invalid status"}})
async def update_incident_status_api(incident_id: str, payload: StatusUpdatePayload):
    new_status = payload.status
    app_logger.info(f"API request to update status for Incident {incident_id} to '{new_status}'")
    incident = incident_store.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")
    
    allowed_statuses = ["Active", "Updated", "Monitoring", "Resolved", "Closed"] 
    if new_status not in allowed_statuses:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status value '{new_status}'. Allowed: {allowed_statuses}")
    try:
        incident_store.update_incident_status(incident_id, new_status)
        return {"message": f"Incident {incident_id} status updated to '{new_status}'."}
    except Exception as e:
         app_logger.error(f"Failed to update status for incident {incident_id} in store: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update incident status.")

@router.delete("/admin/clear_store", summary="Clear all incidents (Admin)", tags=["Admin"], status_code=status.HTTP_200_OK)
async def clear_incident_store():
    app_logger.warning("API request received to clear the entire incident store.")
    try:
        count = len(incident_store.get_all_incidents())
        incident_store.clear_store()
        return {"message": f"Incident store cleared successfully. {count} incidents removed."}
    except Exception as e:
         app_logger.error(f"Failed to clear incident store via API: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear incident store.")