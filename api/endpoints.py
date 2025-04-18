# api/endpoints.py
import logging
from fastapi import APIRouter, HTTPException, Body, Depends, status
from typing import List, Dict, Any
from pydantic import BaseModel, Field # Import BaseModel for request body model

# Import schemas, agent instance, store instance
from data_models.schemas import Incident # Keep Incident for response models
# Remove EidoReport import if not used directly in API signatures
from agent.agent_core import eido_agent_instance
from services.storage import incident_store # Direct access for reads, agent handles writes via process
from config.settings import settings

# Setup logger
logger = logging.getLogger(__name__)
# Configure logging using settings level (ensure settings are loaded first)
# BasicConfig might conflict if uvicorn/FastAPI sets up logging too.
# Consider using FastAPI's dependency injection for logging config if needed.
logging.basicConfig(level=settings.log_level.upper(), format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', force=True)


# Create an API router
router = APIRouter(prefix="/api/v1", tags=["Incidents"]) # Add prefix and tags

# --- Pydantic Model for Alert Text Request ---
class AlertTextPayload(BaseModel):
    alert_text: str = Field(..., description="The raw text content of the alert or report.")

# --- API Endpoints ---

@router.post("/ingest",
             summary="Ingest a single EIDO report (JSON)",
             description="Receives an EIDO report in JSON dictionary format, processes it using the agent, and returns the result.",
             status_code=status.HTTP_201_CREATED,
             response_description="Processing result including incident ID and status message")
async def ingest_eido_report(eido_data: Dict = Body(..., example={
                                                        "eidoMessageIdentifier": "msg_example_123",
                                                        "$id": "msg_example_123",
                                                        "sendingSystemIdentifier": "CADSystemX",
                                                        "lastUpdateTimeStamp": "2024-10-26T10:00:00Z",
                                                        "incidentComponent": [{
                                                            "componentIdentifier": "inc_123",
                                                            "incidentTrackingIdentifier": "FIRE2024-001",
                                                            "lastUpdateTimeStamp": "2024-10-26T10:00:00Z",
                                                            "incidentTypeCommonRegistryText": "Structure Fire",
                                                            "locationReference": "$ref:loc_123"
                                                            }],
                                                        "locationComponent": [{
                                                             "$id": "loc_123",
                                                             "componentIdentifier": "loc_123",
                                                             "locationByValue": "<?xml ...><civicAddressText>123 University Ave</civicAddressText>..."
                                                            }],
                                                        "notesComponent": [{
                                                             "componentIdentifier": "note_123",
                                                             "noteDateTimeStamp": "2024-10-26T10:00:05Z",
                                                             "noteText": "Caller reports smoke..."
                                                            }]
                                                        })):
    """
    Processes an incoming EIDO report provided as a JSON dictionary.
    """
    # Use a more specific ID hint if available
    msg_id_hint = eido_data.get('eidoMessageIdentifier', eido_data.get('$id', 'N/A'))
    logger.info(f"API /ingest received EIDO JSON data (ID hint: {msg_id_hint}).")
    try:
        # Call the agent's JSON processing method
        result_dict = eido_agent_instance.process_report_json(eido_data)

        # Check the result dictionary
        incident_id = result_dict.get('incident_id')
        status_message = result_dict.get('status', 'Processing status unknown')
        is_new = result_dict.get('is_new_incident', False)
        summary = result_dict.get('summary', 'Summary not available') # Get summary from result if available

        if incident_id and status_message.lower() == "success":
            # If successful, return 201 Created
            response_data = {
                "message": "EIDO report processed successfully.",
                "status_detail": status_message,
                "incident_id": incident_id,
                "is_new_incident": is_new,
                "current_summary": summary
            }
            return response_data # FastAPI handles status code 201 based on decorator
        elif status_message.lower().startswith("input error"):
             logger.error(f"API /ingest input error: {status_message}")
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail=f"Failed to process EIDO report: {status_message}"
             )
        else:
            # Handle other processing errors (e.g., LLM failure, matching error)
            logger.error(f"API /ingest processing error: {status_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 422 if validation inside agent failed
                detail=f"Failed to process EIDO report: {status_message}"
            )

    except HTTPException as http_exc:
         raise http_exc # Re-raise existing HTTP exceptions
    except Exception as e:
        logger.critical(f"API /ingest unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred processing the EIDO report: {type(e).__name__}"
        )


# --- NEW ENDPOINT ---
@router.post("/ingest_alert",
             summary="Ingest raw alert text",
             description="Receives raw alert text, uses an LLM to parse it into an EIDO-like structure, processes it using the agent, and returns the result.",
             status_code=status.HTTP_201_CREATED,
             response_description="Processing result including incident ID and status message")
async def ingest_alert_text(payload: AlertTextPayload):
    """
    Processes incoming raw alert text.

    - **payload**: JSON object containing the `alert_text` field.
    """
    alert_text = payload.alert_text
    logger.info(f"API /ingest_alert received raw alert text (Length: {len(alert_text)}).")

    if not alert_text:
         raise HTTPException(
             status_code=status.HTTP_400_BAD_REQUEST,
             detail="Field 'alert_text' cannot be empty."
         )

    try:
        # Call the agent's text processing method
        result_dict = eido_agent_instance.process_alert_text(alert_text)

        # Check the result dictionary
        incident_id = result_dict.get('incident_id')
        status_message = result_dict.get('status', 'Processing status unknown')
        is_new = result_dict.get('is_new_incident', False)
        summary = result_dict.get('summary', 'Summary not available') # Get summary from result

        if incident_id and status_message.lower() == "success":
            # If successful, return 201 Created
            response_data = {
                "message": "Alert text processed successfully.",
                "status_detail": status_message,
                "incident_id": incident_id,
                "is_new_incident": is_new,
                "current_summary": summary
            }
            return response_data # FastAPI handles status code 201
        elif status_message.lower().startswith("input error"):
             logger.error(f"API /ingest_alert input error: {status_message}")
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail=f"Failed to process alert text: {status_message}"
             )
        else:
            # Handle other processing errors (e.g., LLM failure, parsing failure, matching error)
            logger.error(f"API /ingest_alert processing error: {status_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 422 if validation inside agent failed
                detail=f"Failed to process alert text: {status_message}"
            )

    except HTTPException as http_exc:
         raise http_exc # Re-raise existing HTTP exceptions
    except Exception as e:
        logger.critical(f"API /ingest_alert unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred processing the alert text: {type(e).__name__}"
        )


# --- Existing Read Endpoints (keep as they are) ---
@router.get("/incidents",
            response_model=List[Incident],
            summary="List all incidents",
            description="Retrieves a list of all incidents currently tracked by the system.")
async def get_all_incidents():
    """Returns all incidents from the store."""
    logger.info("API request received for /incidents")
    incidents = incident_store.get_all_incidents()
    return incidents

@router.get("/incidents/active",
            response_model=List[Incident],
            summary="List active incidents",
            description="Retrieves incidents that are not in a 'Closed' or 'Resolved' state.")
async def get_active_incidents():
    """Returns active incidents from the store."""
    logger.info("API request received for /incidents/active")
    active_incidents = incident_store.get_active_incidents()
    return active_incidents

@router.get("/incidents/{incident_id}",
            response_model=Incident,
            summary="Get incident details",
            description="Retrieves the full details of a specific incident by its internal ID.",
            responses={404: {"description": "Incident not found"}})
async def get_incident_details(incident_id: str):
    """
    Retrieves details for a specific incident.

    - **incident_id**: The internal unique ID of the incident.
    """
    logger.info(f"API request received for /incidents/{incident_id}")
    incident = incident_store.get_incident(incident_id)
    if incident:
        return incident
    else:
        logger.warning(f"API request for non-existent Incident ID: {incident_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found."
        )

@router.get("/incidents/{incident_id}/summary",
            summary="Get incident summary",
            response_description="The current summary of the incident",
            responses={404: {"description": "Incident not found"}})
async def get_incident_summary_api(incident_id: str):
    """Returns just the summary for a specific incident."""
    logger.info(f"API request received for /incidents/{incident_id}/summary")
    incident = incident_store.get_incident(incident_id)
    if incident:
        return {"incident_id": incident_id, "summary": incident.summary}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")

@router.get("/incidents/{incident_id}/recommendations",
            summary="Get recommended actions",
            response_description="List of recommended actions for the incident",
            responses={404: {"description": "Incident not found"}})
async def get_incident_recommendations_api(incident_id: str):
    """Returns the recommended actions for a specific incident."""
    logger.info(f"API request received for /incidents/{incident_id}/recommendations")
    incident = incident_store.get_incident(incident_id)
    if incident:
        return {"incident_id": incident_id, "recommended_actions": incident.recommended_actions}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")

# --- Admin Endpoints ---
@router.put("/incidents/{incident_id}/status",
            summary="Update incident status",
            status_code=status.HTTP_200_OK,
            response_description="Confirmation of status update",
            responses={404: {"description": "Incident not found"}, 400: {"description": "Invalid status"}},
            tags=["Admin"]) # Add tag
async def update_incident_status_api(incident_id: str, status_update: Dict[str, str] = Body(..., example={"status": "Resolved"})):
    """Updates the status of an incident (e.g., 'Active', 'Closed', 'Resolved')."""
    new_status = status_update.get("status")
    if not new_status:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'status' field in request body.")

    logger.info(f"API request to update status for Incident {incident_id} to '{new_status}'")
    incident = incident_store.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")

    allowed_statuses = ["Active", "Updated", "Monitoring", "Resolved", "Closed"] # Keep allowed statuses
    if new_status not in allowed_statuses:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status value '{new_status}'. Allowed values: {allowed_statuses}")

    try:
        incident_store.update_incident_status(incident_id, new_status)
        return {"message": f"Incident {incident_id} status updated to '{new_status}'."}
    except Exception as e:
         logger.error(f"Failed to update status for incident {incident_id} in store: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update incident status in storage.")


@router.delete("/admin/clear_store",
            summary="Clear all incidents (Admin)",
            status_code=status.HTTP_200_OK,
            response_description="Confirmation that the store is cleared",
            include_in_schema=True, # Keep visible for testing
            tags=["Admin"]) # Add tag
async def clear_incident_store():
    """ADMIN ONLY: Clears all incidents from the in-memory store."""
    logger.warning("API request received to clear the entire incident store.")
    try:
        count = len(incident_store.get_all_incidents())
        incident_store.clear_store()
        return {"message": f"Incident store cleared successfully. {count} incidents removed."}
    except Exception as e:
         logger.error(f"Failed to clear incident store via API: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear incident store.")