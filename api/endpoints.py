# api/endpoints.py
import logging
from fastapi import APIRouter, HTTPException, Body, Depends, status
from typing import List, Dict, Any

# Import schemas, agent instance, store instance
from data_models.schemas import Incident, EidoReport # Import base model for request/response typing
from agent.agent_core import eido_agent_instance
from services.storage import incident_store # Direct access for reads, agent handles writes via process
from config.settings import settings

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level.upper())

# Create an API router
router = APIRouter()

# --- API Endpoints ---

@router.post("/ingest",
             summary="Ingest a single EIDO report",
             description="Receives an EIDO report in JSON format, processes it using the agent (matching, summarization, recommendations), and returns the result.",
             status_code=status.HTTP_201_CREATED, # Use 201 for successful creation/update trigger
             response_description="Processing result including incident ID and status message")
async def ingest_eido_report(report_data: Dict = Body(..., example={
                                                        "report_id": "CAD12345_Example",
                                                        "incident_id_external": "FIRE2024-001",
                                                        "timestamp": "2024-10-26T10:00:00Z",
                                                        "incident_type": "Structure Fire",
                                                        "description": "Caller reports smoke...",
                                                        "location": {"address": "123 University Ave"},
                                                        "source": "CAD"
                                                        })):
    """
    Processes an incoming EIDO report.

    - **report_data**: The EIDO report content as a JSON object.
    """
    logger.info(f"API /ingest received report data (ID hint: {report_data.get('report_id', 'N/A')}).")
    incident_id, is_new, status_message = eido_agent_instance.process_report_json(report_data)

    if incident_id:
        # Retrieve the potentially updated incident to include its summary in the response
        incident = incident_store.get_incident(incident_id)
        current_summary = incident.summary if incident else "Summary not available post-processing."
        response_data = {
            "message": status_message,
            "incident_id": incident_id,
            "is_new_incident": is_new,
            "current_summary": current_summary
        }
        # Use 201 if new, 200 if update? Or consistently 201 as resource state changed. Let's stick to 201.
        return response_data
    else:
        # If processing failed entirely (e.g., parsing error), return 400 or 500
        # Status message should contain the error details
        logger.error(f"API /ingest failed processing: {status_message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # Assuming input error mostly
            detail=f"Failed to process report: {status_message}"
        )

@router.get("/incidents",
            response_model=List[Incident], # Ensure output matches the Incident schema
            summary="List all incidents",
            description="Retrieves a list of all incidents currently tracked by the system.")
async def get_all_incidents():
    """Returns all incidents from the store."""
    logger.info("API request received for /incidents")
    incidents = incident_store.get_all_incidents()
    # Pydantic V2 + FastAPI usually handle datetime serialization well.
    # If issues arise, manual conversion might be needed before returning.
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

# Example: Endpoint to change incident status (if needed)
@router.put("/incidents/{incident_id}/status",
            summary="Update incident status",
            status_code=status.HTTP_200_OK,
            response_description="Confirmation of status update",
            responses={404: {"description": "Incident not found"}, 400: {"description": "Invalid status"}})
async def update_incident_status_api(incident_id: str, status_update: Dict[str, str] = Body(..., example={"status": "Resolved"})):
    """Updates the status of an incident (e.g., 'Active', 'Closed', 'Resolved')."""
    new_status = status_update.get("status")
    if not new_status:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'status' field in request body.")

    logger.info(f"API request to update status for Incident {incident_id} to '{new_status}'")
    incident = incident_store.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident with ID '{incident_id}' not found.")

    # Optional: Validate the new status against allowed values
    allowed_statuses = ["Active", "Updated", "Monitoring", "Resolved", "Closed"]
    if new_status not in allowed_statuses:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status value '{new_status}'. Allowed values: {allowed_statuses}")

    incident_store.update_incident_status(incident_id, new_status)
    return {"message": f"Incident {incident_id} status updated to '{new_status}'."}


# Example: Endpoint to clear the store (for development/testing)
@router.delete("/admin/clear_store",
            summary="Clear all incidents (Admin)",
            status_code=status.HTTP_200_OK,
            response_description="Confirmation that the store is cleared",
            include_in_schema=True) # Set to False to hide from public docs
async def clear_incident_store():
    """ADMIN ONLY: Clears all incidents from the in-memory store."""
    logger.warning("API request received to clear the entire incident store.")
    count = len(incident_store.get_all_incidents())
    incident_store.clear_store()
    return {"message": f"Incident store cleared successfully. {count} incidents removed."}