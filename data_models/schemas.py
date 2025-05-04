# data_models/schemas.py
import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)

# --- Core Data Extracted from Individual Reports ---

class ReportCoreData(BaseModel):
    """ Simplified representation of essential info from a single report. """
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    external_incident_id: Optional[str] = Field(None)
    timestamp: datetime
    incident_type: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    location_address: Optional[str] = Field(None)
    coordinates: Optional[Tuple[float, float]] = Field(None)
    # --- NEW FIELD ---
    zip_code: Optional[str] = Field(None, description="Postal/ZIP code associated with the report.")
    # --- END NEW FIELD ---
    source: Optional[str] = Field(None)
    original_document_id: Optional[str] = Field(None)
    original_eido_dict: Optional[Dict[str, Any]] = Field(None)

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_timezone_aware(cls, v):
        # (Keep existing implementation)
        if isinstance(v, str):
            try:
                dt = datetime.fromisoformat(v.replace('Z', '+00:00')) # Handle Z properly
                if dt.tzinfo is None: return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError: return datetime.now(timezone.utc)
        elif isinstance(v, datetime):
            if v.tzinfo is None: return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    model_config = ConfigDict(extra='allow')


class Incident(BaseModel):
    """ Consolidated view of an emergency incident. """
    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_type: Optional[str] = Field(None)
    status: str = Field("Active")
    created_at: Optional[datetime] = Field(None)
    last_updated_at: Optional[datetime] = Field(None)
    summary: str = Field("Summary not yet generated.")
    recommended_actions: List[str] = Field(default_factory=list)
    reports_core_data: List[ReportCoreData] = Field(default_factory=list)
    locations: List[Tuple[float, float]] = Field(default_factory=list) # Unique coordinates
    addresses: List[str] = Field(default_factory=list) # Unique addresses
    # --- NEW FIELD ---
    zip_codes: List[str] = Field(default_factory=list, description="List of unique ZIP codes associated with the incident.")
    # --- END NEW FIELD ---
    trend_data: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra='allow')

    def add_report_core_data(self, core_data: ReportCoreData, match_info: Optional[str] = None):
        """Adds core data, updates timestamps, locations, addresses, zip codes, trends."""
        if not isinstance(core_data, ReportCoreData):
            logger.error(f"Invalid type added to Incident {self.incident_id[:8]}: {type(core_data)}")
            return

        logger.debug(f"Adding Report {core_data.report_id[:8]} to Incident {self.incident_id[:8]}.")
        self.reports_core_data.append(core_data)

        # Update timestamps
        report_ts = core_data.timestamp # Assumed UTC
        if self.created_at is None or report_ts < self.created_at: self.created_at = report_ts
        if self.last_updated_at is None or report_ts > self.last_updated_at: self.last_updated_at = report_ts

        # Update incident type (simple logic)
        if core_data.incident_type and (not self.incident_type or self.incident_type == "Unknown"):
             self.incident_type = core_data.incident_type

        # Update unique locations
        if core_data.coordinates and isinstance(core_data.coordinates, tuple) and len(core_data.coordinates) == 2:
            try:
                coords_tuple = (float(core_data.coordinates[0]), float(core_data.coordinates[1]))
                if coords_tuple not in self.locations: self.locations.append(coords_tuple)
            except (ValueError, TypeError): logger.warning(f"Invalid coords format in report {core_data.report_id[:8]}: {core_data.coordinates}")

        # Update unique addresses
        if core_data.location_address and core_data.location_address not in self.addresses:
            self.addresses.append(core_data.location_address)

        # --- Update unique zip codes ---
        if core_data.zip_code and isinstance(core_data.zip_code, str) and core_data.zip_code not in self.zip_codes:
            self.zip_codes.append(core_data.zip_code)
            logger.debug(f"Added unique ZIP code '{core_data.zip_code}' to Incident {self.incident_id[:8]}.")
        # --- End update ---

        # Update trend data
        self.trend_data['report_count'] = len(self.reports_core_data)
        if self.created_at and self.last_updated_at:
             duration_seconds = (self.last_updated_at - self.created_at).total_seconds()
             self.trend_data['duration_minutes'] = round(duration_seconds / 60.0, 1)
        if match_info: self.trend_data['match_info'] = match_info

        # Update status
        if len(self.reports_core_data) > 1 and self.status == "Active": self.status = "Updated"

        logger.info(f"Incident {self.incident_id[:8]} updated by Report {core_data.report_id[:8]}. Reports: {self.trend_data['report_count']}.")

    def get_full_description_history(self, exclude_latest=False) -> str:
        # (Keep existing implementation)
        history_entries = []
        reports_to_include = self.reports_core_data[:-1] if exclude_latest else self.reports_core_data
        try: sorted_reports = sorted(reports_to_include, key=lambda r: r.timestamp if r.timestamp else datetime.min.replace(tzinfo=timezone.utc))
        except Exception as sort_e: logger.warning(f"Error sorting reports history: {sort_e}"); sorted_reports = reports_to_include
        for report in sorted_reports:
            if report.description:
                ts_str = report.timestamp.strftime('%Y-%m-%d %H:%M Z') if report.timestamp else "Unknown Time"
                source_str = f" (Src: {report.source})" if report.source else ""
                history_entries.append(f"[{ts_str}{source_str}]: {report.description}")
        return "\n---\n".join(history_entries) if history_entries else "No description history."