# services/storage.py
import logging
import uuid
from typing import Dict, List, Optional

# Ensure Incident and ReportCoreData schemas are correctly imported
# Assuming they are defined as shown previously in data_models/schemas.py
try:
    from data_models.schemas import Incident, ReportCoreData
except ImportError:
    # Handle potential import issues if structure is different
    print("ERROR: Failed to import Incident or ReportCoreData from data_models.schemas")
    # Define dummy classes or raise error to prevent running with incorrect setup
    class Incident: pass
    class ReportCoreData: pass
    # raise # Or raise the ImportError

logger = logging.getLogger(__name__)

class IncidentStore:
    """
    In-memory storage for consolidated Incident objects.
    Uses a dictionary with incident_id as the key.
    NOTE: This is non-persistent and will be lost when the application stops.
    """
    def __init__(self):
        """Initializes an empty incident store."""
        self.incidents: Dict[str, Incident] = {}
        logger.info("In-memory IncidentStore initialized.")

    def create_new_incident_from_core_data(self, core_data: ReportCoreData, summary: str, recommendations: List[str]) -> Incident:
        """
        Creates a new Incident object from the first ReportCoreData,
        populates it, adds it to the store, and returns it.
        """
        # Create the Incident instance first
        new_incident = Incident(
            incident_id=str(uuid.uuid4()), # Generate the unique ID here
            summary=summary,
            recommended_actions=recommendations,
            status="Active" # Set initial status explicitly
        )

        # Set initial values based on the first report's core data
        new_incident.incident_type = core_data.incident_type
        # created_at and last_updated_at will be set by add_report_core_data based on the first report's timestamp
        # Ensure the timestamp is correctly passed and handled in Incident.add_report_core_data

        # Add the core data - this populates reports_core_data, locations,
        # updates last_updated_at, report_count etc.
        # It should also set created_at based on the first report added.
        try:
             new_incident.add_report_core_data(core_data)
             # Explicitly set created_at if add_report_core_data doesn't handle it for the very first report
             if not new_incident.created_at or new_incident.trend_data.get('report_count', 0) <= 1:
                 new_incident.created_at = core_data.timestamp
        except Exception as e:
             logger.error(f"Error adding initial core data to new incident {new_incident.incident_id[:8]}: {e}", exc_info=True)
             # Decide if the incident should still be stored or if this is critical
             # For now, store it even if add_report_core_data failed partially

        # Add the fully populated incident to the store
        self.incidents[new_incident.incident_id] = new_incident
        logger.info(f"Created and stored new Incident {new_incident.incident_id[:8]} (Type: {new_incident.incident_type}) from Report {core_data.report_id[:8]}.")
        return new_incident

    def save_incident(self, incident: Incident):
        """
        Explicitly saves or updates the incident object in the store.
        Uses the incident's incident_id as the key.
        """
        if not hasattr(incident, 'incident_id') or not incident.incident_id:
             logger.error("Attempted to save an incident without an incident_id!")
             return # Cannot save without an ID

        is_update = incident.incident_id in self.incidents
        self.incidents[incident.incident_id] = incident
        action = "Updated" if is_update else "Saved (new)"
        logger.debug(f"{action} Incident {incident.incident_id[:8]} in store.")

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Retrieves a single incident by its full ID."""
        incident = self.incidents.get(incident_id)
        if incident:
            logger.debug(f"Retrieved Incident {incident_id[:8]} from store.")
        else:
            logger.warning(f"Incident ID {incident_id} not found in store.")
        return incident

    def get_all_incidents(self) -> List[Incident]:
         """Returns a list of all incident objects currently in the store."""
         all_incs = list(self.incidents.values())
         logger.debug(f"Retrieved all {len(all_incs)} incidents from store.")
         return all_incs

    def get_active_incidents(self) -> List[Incident]:
          """
          Returns a list of incidents considered 'active' based on status.
          Adjust the status list as needed for your definition of active.
          """
          # Define statuses considered active (case-insensitive check)
          active_statuses = [
              "active", "updated", "received", "rcvd",
              "dispatched", "dsp", "acknowledged", "ack",
              "enroute", "enr", "onscene", "onscn"
          ]
          active_incs = [
              inc for inc in self.incidents.values()
              if inc.status and inc.status.lower() in active_statuses
          ]
          logger.debug(f"Found {len(active_incs)} active incidents out of {len(self.incidents)} total.")
          return active_incs

    def clear_store(self):
          """Removes all incidents from the in-memory store."""
          count = len(self.incidents)
          self.incidents.clear()
          logger.warning(f"Cleared {count} incidents from the store. Store is now empty.")


# --- Singleton Instance ---
# Create a single instance of the store to be used across the application
incident_store = IncidentStore()