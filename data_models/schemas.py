# data_models/schemas.py
import logging
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import uuid
from collections import Counter

logger = logging.getLogger(__name__)

# --- Core Data Extracted from Individual Reports ---

class ReportCoreData(BaseModel):
    """
    A simplified, flattened representation of the essential information
    extracted from a single incoming EIDO report/message.
    This is used as input for matching and updating incidents.
    """
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique internal ID for this processed report data.")
    external_incident_id: Optional[str] = Field(None, description="Incident Tracking ID from the source system (e.g., CAD ID).")
    timestamp: datetime = Field(..., description="Timestamp associated with the report (e.g., last update time).")
    incident_type: Optional[str] = Field(None, description="Type of incident (e.g., 'Traffic Collision', 'Structure Fire').")
    description: Optional[str] = Field(None, description="Narrative or description text from the report.")
    location_address: Optional[str] = Field(None, description="Civic address associated with the report.")
    coordinates: Optional[Tuple[float, float]] = Field(None, description="Latitude and Longitude tuple.")
    source: Optional[str] = Field(None, description="Originating system or agency identifier.")
    original_document_id: Optional[str] = Field(None, description="Identifier of the original EIDO message (e.g., eidoMessageIdentifier).")
    # --- >>> NEW FIELD <<< ---
    original_eido_dict: Optional[Dict[str, Any]] = Field(None, description="The original EIDO JSON dictionary this core data was extracted from.")
    # --- >>> END NEW FIELD <<< ---


    # Ensure timestamps are timezone-aware (UTC)
    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_timezone_aware(cls, v):
        if isinstance(v, str):
            try:
                dt = datetime.fromisoformat(v)
                if dt.tzinfo is None:
                    # logger.warning(f"Timestamp '{v}' was naive, assuming UTC.") # Reduce noise
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                logger.error(f"Could not parse timestamp string: {v}. Using current UTC time.")
                return datetime.now(timezone.utc) # Fallback
        elif isinstance(v, datetime):
            if v.tzinfo is None:
                # logger.warning(f"Timestamp '{v}' was naive, assuming UTC.") # Reduce noise
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        logger.error(f"Unexpected type for timestamp: {type(v)}. Using current UTC time.")
        return datetime.now(timezone.utc) # Fallback

    model_config = ConfigDict(extra='allow') # Allow extra fields if needed internally


# --- Consolidated Incident Object ---
# (Incident class remains the same, no changes needed here)
class Incident(BaseModel):
    """
    Represents a consolidated view of an emergency incident, potentially
    synthesized from multiple EIDO reports over time.
    """
    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique internal identifier for the consolidated incident.")
    incident_type: Optional[str] = Field(None, description="The determined type of the incident.")
    status: str = Field("Active", description="Current status of the incident (e.g., Active, Updated, Closed).")
    created_at: Optional[datetime] = Field(None, description="Timestamp when the incident was first created in this system.")
    last_updated_at: Optional[datetime] = Field(None, description="Timestamp when the incident was last updated by a new report.")

    summary: str = Field("Summary not yet generated.", description="AI-generated summary of the incident's current state.")
    recommended_actions: List[str] = Field(default_factory=list, description="AI-generated list of recommended next actions.")

    # Store the history of core data from each report contributing to this incident
    reports_core_data: List[ReportCoreData] = Field(default_factory=list, description="List of core data extracted from each contributing report.")

    # Store unique locations associated with this incident
    locations: List[Tuple[float, float]] = Field(default_factory=list, description="List of unique geographic coordinates associated with the incident.")
    addresses: List[str] = Field(default_factory=list, description="List of unique addresses associated with the incident.")

    # Store trend/meta data
    trend_data: Dict[str, Any] = Field(default_factory=dict, description="Dictionary to store trend analysis data (e.g., report count, duration, match info).")

    # Allow extra fields during model creation/validation if needed
    model_config = ConfigDict(extra='allow')

    def add_report_core_data(self, core_data: ReportCoreData, match_info: Optional[str] = None):
        """Adds core data from a new report, updates timestamps, locations, and trend data."""
        if not isinstance(core_data, ReportCoreData):
            logger.error(f"Attempted to add non-ReportCoreData object to Incident {self.incident_id[:8]}. Type: {type(core_data)}")
            return

        logger.debug(f"Adding Report {core_data.report_id[:8]} to Incident {self.incident_id[:8]}.")
        self.reports_core_data.append(core_data)

        # Update timestamps
        report_ts = core_data.timestamp # Already validated to be timezone-aware UTC
        if self.created_at is None or report_ts < self.created_at:
            self.created_at = report_ts
        if self.last_updated_at is None or report_ts > self.last_updated_at:
            self.last_updated_at = report_ts

        # Update incident type if not set or if new report has a more specific one?
        # Simple logic: take the first non-null type, or the latest non-null type.
        if core_data.incident_type and (not self.incident_type or self.incident_type == "Unknown"):
             self.incident_type = core_data.incident_type
             logger.debug(f"Incident {self.incident_id[:8]} type updated to '{self.incident_type}' from report {core_data.report_id[:8]}.")

        # Update unique locations and addresses
        if core_data.coordinates:
            if isinstance(core_data.coordinates, tuple) and len(core_data.coordinates) == 2 and all(isinstance(c, (float, int)) for c in core_data.coordinates):
                coords_tuple = (float(core_data.coordinates[0]), float(core_data.coordinates[1]))
                if coords_tuple not in self.locations:
                    self.locations.append(coords_tuple)
                    # logger.debug(f"Added unique location {coords_tuple} to Incident {self.incident_id[:8]}.") # Reduce noise
            else:
                 logger.warning(f"Report {core_data.report_id[:8]} had invalid coordinates format: {core_data.coordinates}. Not added to incident locations.")

        if core_data.location_address and core_data.location_address not in self.addresses:
            self.addresses.append(core_data.location_address)
            # logger.debug(f"Added unique address '{core_data.location_address}' to Incident {self.incident_id[:8]}.") # Reduce noise

        # Update trend data
        self.trend_data['report_count'] = len(self.reports_core_data)
        if self.created_at and self.last_updated_at:
             duration_seconds = (self.last_updated_at - self.created_at).total_seconds()
             self.trend_data['duration_minutes'] = round(duration_seconds / 60.0, 1)
        if match_info:
             self.trend_data['match_info'] = match_info # Store how the last report was matched

        # Update status (simple logic: mark as 'Updated' if not new)
        if len(self.reports_core_data) > 1:
            self.status = "Updated"

        logger.info(f"Incident {self.incident_id[:8]} updated with Report {core_data.report_id[:8]}. Total reports: {self.trend_data['report_count']}.")


    def get_full_description_history(self, exclude_latest=False) -> str:
        """
        Constructs a chronological history of descriptions from associated reports.
        """
        history_entries = []
        reports_to_include = self.reports_core_data[:-1] if exclude_latest else self.reports_core_data

        # Sort by timestamp just in case they weren't added chronologically
        try:
            sorted_reports = sorted(
                reports_to_include,
                key=lambda r: r.timestamp if r.timestamp else datetime.min.replace(tzinfo=timezone.utc) # Handle None ts
            )
        except Exception as sort_e:
            logger.warning(f"Error sorting reports for history: {sort_e}")
            sorted_reports = reports_to_include # fallback to original order

        for report in sorted_reports:
            if report.description:
                ts_str = report.timestamp.strftime('%Y-%m-%d %H:%M:%S Z') if report.timestamp else "Unknown Time"
                source_str = f" (Source: {report.source})" if report.source else ""
                history_entries.append(f"[{ts_str}{source_str}]: {report.description}")

        return "\n---\n".join(history_entries) if history_entries else "No description history available."