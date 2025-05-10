import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)

class ReportCoreData(BaseModel):
    """ Simplified representation of essential info from a single report. """
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    external_incident_id: Optional[str] = Field(None, description="External ID like CAD number.")
    timestamp: datetime
    incident_type: Optional[str] = Field(None, description="Type of incident, e.g., 'Structure Fire'.")
    description: Optional[str] = Field(None, description="Narrative or notes of the report.")
    location_address: Optional[str] = Field(None, description="Full street address or intersection.")
    coordinates: Optional[Tuple[float, float]] = Field(None, description="Geographic coordinates (latitude, longitude).")
    zip_code: Optional[str] = Field(None, description="Postal/ZIP code associated with the report.")
    source: Optional[str] = Field(None, description="Reporting source, e.g., agency name, system ID.")
    original_document_id: Optional[str] = Field(None, description="ID of the original EIDO message or source document.")
    original_eido_dict: Optional[Dict[str, Any]] = Field(None, description="The raw EIDO dictionary if source was JSON.")

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_timezone_aware(cls, v):
        if isinstance(v, str):
            try:
                # Handle 'Z' for UTC and timestamps with existing offsets
                if v.endswith('Z'): dt = datetime.fromisoformat(v[:-1] + '+00:00')
                else: dt = datetime.fromisoformat(v)
                
                if dt.tzinfo is None: return dt.replace(tzinfo=timezone.utc) # Assume UTC if naive
                return dt.astimezone(timezone.utc) # Convert to UTC
            except ValueError as e:
                logger.warning(f"Could not parse timestamp string '{v}': {e}. Using current UTC time.")
                return datetime.now(timezone.utc)
        elif isinstance(v, datetime):
            if v.tzinfo is None: return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        logger.warning(f"Invalid timestamp type '{type(v)}': {v}. Using current UTC time.")
        return datetime.now(timezone.utc)

    model_config = ConfigDict(extra='allow', validate_assignment=True)


class Incident(BaseModel):
    """ Consolidated view of an emergency incident. """
    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_type: Optional[str] = Field(None)
    status: str = Field("Active", description="Current status, e.g., Active, Updated, Resolved, Closed.")
    created_at: Optional[datetime] = Field(None)
    last_updated_at: Optional[datetime] = Field(None)
    summary: str = Field("Summary not yet generated.")
    recommended_actions: List[str] = Field(default_factory=list)
    
    reports_core_data: List[ReportCoreData] = Field(default_factory=list)
    
    locations: List[Tuple[float, float]] = Field(default_factory=list, description="Unique (lat,lon) coordinates associated.")
    addresses: List[str] = Field(default_factory=list, description="Unique textual addresses associated.")
    zip_codes: List[str] = Field(default_factory=list, description="List of unique ZIP codes associated.")
    
    trend_data: Dict[str, Any] = Field(default_factory=dict, description="Metrics like report count, duration, match info.")
    model_config = ConfigDict(extra='allow', validate_assignment=True)

    def add_report_core_data(self, core_data: ReportCoreData, match_info: Optional[str] = None):
        if not isinstance(core_data, ReportCoreData):
            logger.error(f"Invalid type added to Incident {self.incident_id[:8]}: {type(core_data)}")
            return

        logger.debug(f"Adding Report {core_data.report_id[:8]} (ExtID: {core_data.external_incident_id or 'N/A'}) to Incident {self.incident_id[:8]}.")
        self.reports_core_data.append(core_data)

        report_ts = core_data.timestamp # Already timezone-aware UTC from ReportCoreData
        if self.created_at is None or report_ts < self.created_at: self.created_at = report_ts
        if self.last_updated_at is None or report_ts > self.last_updated_at: self.last_updated_at = report_ts

        if core_data.incident_type and (not self.incident_type or self.incident_type == "Unknown" or self.incident_type == "Unknown - Parsed Alert"):
             self.incident_type = core_data.incident_type

        if core_data.coordinates and isinstance(core_data.coordinates, tuple) and len(core_data.coordinates) == 2:
            try:
                coords_tuple = (float(core_data.coordinates[0]), float(core_data.coordinates[1]))
                if coords_tuple not in self.locations: self.locations.append(coords_tuple)
            except (ValueError, TypeError): logger.warning(f"Invalid coords format in report {core_data.report_id[:8]}: {core_data.coordinates}")

        if core_data.location_address and core_data.location_address not in self.addresses:
            self.addresses.append(core_data.location_address)

        if core_data.zip_code and isinstance(core_data.zip_code, str) and core_data.zip_code not in self.zip_codes:
            self.zip_codes.append(core_data.zip_code)
            logger.debug(f"Added unique ZIP code '{core_data.zip_code}' to Incident {self.incident_id[:8]}.")

        self.trend_data['report_count'] = len(self.reports_core_data)
        if self.created_at and self.last_updated_at:
             duration_seconds = (self.last_updated_at - self.created_at).total_seconds()
             self.trend_data['duration_minutes'] = round(duration_seconds / 60.0, 1)
        if match_info: self.trend_data['last_match_info'] = match_info # Store latest match info

        # Sensible status update: if it was 'Active' and gets a new report, it becomes 'Updated'.
        # More complex status logic (e.g., based on EIDO status fields) could be added here or in agent_core.
        if len(self.reports_core_data) > 1 and self.status == "Active":
            self.status = "Updated"
        
        # If this is the very first report, ensure created_at is set from this report's timestamp
        if len(self.reports_core_data) == 1:
             self.created_at = core_data.timestamp
             self.last_updated_at = core_data.timestamp # Also init last_updated_at

        logger.info(f"Incident {self.incident_id[:8]} updated by Report {core_data.report_id[:8]}. Reports: {self.trend_data.get('report_count',0)}, Status: {self.status}.")

    def get_full_description_history(self, exclude_latest: bool = False) -> str:
        history_entries = []
        # Sort reports by timestamp before creating history
        try:
            # Ensure timestamps are valid before sorting
            valid_reports = [r for r in self.reports_core_data if r.timestamp is not None]
            sorted_reports = sorted(valid_reports, key=lambda r: r.timestamp)
            if exclude_latest and sorted_reports:
                reports_to_include = sorted_reports[:-1]
            else:
                reports_to_include = sorted_reports
        except Exception as sort_e:
            logger.warning(f"Error sorting reports for history: {sort_e}. Using original order.", exc_info=True)
            reports_to_include = self.reports_core_data[:-1] if exclude_latest and self.reports_core_data else self.reports_core_data

        for report in reports_to_include:
            if report.description:
                # Ensure timestamp is datetime before strftime
                ts_str = "Unknown Time"
                if isinstance(report.timestamp, datetime):
                    ts_str = report.timestamp.strftime('%Y-%m-%d %H:%M:%S Z')
                
                source_str = f" (Src: {report.source or 'N/A'})"
                ext_id_str = f" (ExtID: {report.external_incident_id or 'N/A'})"
                history_entries.append(f"[{ts_str}{source_str}{ext_id_str}]: {report.description}")
        return "\n---\n".join(history_entries) if history_entries else "No description history available."