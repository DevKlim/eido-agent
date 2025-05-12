import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
import json # For casting some ReportCoreData fields before saving
from sqlalchemy import delete

from data_models.schemas import Incident as PydanticIncident, ReportCoreData as PydanticReportCoreData
from services.database import get_db_session, IncidentDB, ReportCoreDataDB, AsyncSessionLocal

logger = logging.getLogger(__name__)

class IncidentStore:
    def __init__(self):
        logger.info("Database-backed IncidentStore initialized.")

    async def _pydantic_to_incident_db(self, p_incident: PydanticIncident) -> IncidentDB:
        # Convert lists/tuples to JSON-serializable format if storing directly in JSONB
        # For ReportCoreData, we'll handle its conversion when saving related reports
        return IncidentDB(
            id=uuid.UUID(p_incident.incident_id) if isinstance(p_incident.incident_id, str) else p_incident.incident_id,
            incident_type=p_incident.incident_type,
            status=p_incident.status,
            created_at=p_incident.created_at,
            last_updated_at=p_incident.last_updated_at,
            summary=p_incident.summary,
            recommended_actions=p_incident.recommended_actions, # Already list of strings
            locations_coords=[list(loc) if isinstance(loc, tuple) else loc for loc in p_incident.locations], # Ensure list of lists for JSON
            addresses=p_incident.addresses,
            zip_codes=p_incident.zip_codes,
            trend_data=p_incident.trend_data
        )

    async def _incident_db_to_pydantic(self, db_incident: IncidentDB, reports_core_data: List[PydanticReportCoreData]) -> PydanticIncident:
        return PydanticIncident(
            incident_id=str(db_incident.id),
            incident_type=db_incident.incident_type,
            status=db_incident.status,
            created_at=db_incident.created_at,
            last_updated_at=db_incident.last_updated_at,
            summary=db_incident.summary,
            recommended_actions=db_incident.recommended_actions if isinstance(db_incident.recommended_actions, list) else [],
            locations=[tuple(loc) if isinstance(loc, list) else loc for loc in (db_incident.locations_coords or [])], # Convert back to tuples
            addresses=db_incident.addresses if isinstance(db_incident.addresses, list) else [],
            zip_codes=db_incident.zip_codes if isinstance(db_incident.zip_codes, list) else [],
            trend_data=db_incident.trend_data if isinstance(db_incident.trend_data, dict) else {},
            reports_core_data=reports_core_data
        )

    async def _pydantic_to_report_core_db(self, p_report: PydanticReportCoreData, incident_id_uuid: uuid.UUID) -> ReportCoreDataDB:
        coords_lat, coords_lon = (p_report.coordinates[0], p_report.coordinates[1]) if p_report.coordinates else (None, None)
        
        # Ensure original_eido_dict is serializable; Pydantic might handle this, but explicit check is safer
        original_eido_dict_serializable = None
        if p_report.original_eido_dict:
            try:
                # Test serialization (though SQLAlchemy JSONB handles dicts directly)
                json.dumps(p_report.original_eido_dict) 
                original_eido_dict_serializable = p_report.original_eido_dict
            except TypeError:
                logger.warning(f"original_eido_dict for report {p_report.report_id} is not JSON serializable. Storing as string representation.")
                original_eido_dict_serializable = {"error": "Unserializable data", "content_str": str(p_report.original_eido_dict)}


        return ReportCoreDataDB(
            id=uuid.UUID(p_report.report_id) if isinstance(p_report.report_id, str) else p_report.report_id,
            incident_id=incident_id_uuid,
            external_incident_id=p_report.external_incident_id,
            timestamp=p_report.timestamp,
            incident_type=p_report.incident_type,
            description=p_report.description,
            location_address=p_report.location_address,
            coordinates_lat=coords_lat,
            coordinates_lon=coords_lon,
            zip_code=p_report.zip_code,
            source=p_report.source,
            original_document_id=p_report.original_document_id,
            original_eido_dict=original_eido_dict_serializable # Should be a dict
        )

    async def _report_core_db_to_pydantic(self, db_report: ReportCoreDataDB) -> PydanticReportCoreData:
        coords = (db_report.coordinates_lat, db_report.coordinates_lon) if db_report.coordinates_lat is not None and db_report.coordinates_lon is not None else None
        return PydanticReportCoreData(
            report_id=str(db_report.id),
            external_incident_id=db_report.external_incident_id,
            timestamp=db_report.timestamp,
            incident_type=db_report.incident_type,
            description=db_report.description,
            location_address=db_report.location_address,
            coordinates=coords, # type: ignore
            zip_code=db_report.zip_code,
            source=db_report.source,
            original_document_id=db_report.original_document_id,
            original_eido_dict=db_report.original_eido_dict if isinstance(db_report.original_eido_dict, dict) else {}
        )

    async def save_incident(self, p_incident: PydanticIncident):
        async with get_db_session() as session:
            incident_id_uuid = uuid.UUID(p_incident.incident_id) if isinstance(p_incident.incident_id, str) else p_incident.incident_id
            
            # Check if incident exists
            result = await session.execute(select(IncidentDB).where(IncidentDB.id == incident_id_uuid))
            db_incident = result.scalars().first()

            if db_incident: # Update existing incident
                db_incident.incident_type = p_incident.incident_type
                db_incident.status = p_incident.status
                db_incident.created_at = p_incident.created_at
                db_incident.last_updated_at = p_incident.last_updated_at
                db_incident.summary = p_incident.summary
                db_incident.recommended_actions = p_incident.recommended_actions
                db_incident.locations_coords = [list(loc) if isinstance(loc, tuple) else loc for loc in p_incident.locations]
                db_incident.addresses = p_incident.addresses
                db_incident.zip_codes = p_incident.zip_codes
                db_incident.trend_data = p_incident.trend_data
                logger.debug(f"Updating Incident {p_incident.incident_id[:8]} in DB.")
            else: # Create new incident
                db_incident = await self._pydantic_to_incident_db(p_incident)
                session.add(db_incident)
                logger.debug(f"Saving new Incident {p_incident.incident_id[:8]} to DB.")

            # Handle reports_core_data: delete existing for this incident and re-add all from PydanticIncident
            # This is a simple way to sync; more complex merging could be done.
            await session.execute(delete(ReportCoreDataDB).where(ReportCoreDataDB.incident_id == incident_id_uuid)) # type: ignore
            
            for p_report in p_incident.reports_core_data:
                db_report = await self._pydantic_to_report_core_db(p_report, incident_id_uuid)
                session.add(db_report)
            
            await session.commit()
            logger.info(f"Saved Incident {p_incident.incident_id[:8]} with {len(p_incident.reports_core_data)} reports to DB.")

    async def get_incident(self, incident_id_str: str) -> Optional[PydanticIncident]:
        async with get_db_session() as session:
            try:
                incident_id_uuid = uuid.UUID(incident_id_str)
            except ValueError:
                logger.warning(f"Invalid UUID format for incident_id: {incident_id_str}")
                return None

            result = await session.execute(select(IncidentDB).where(IncidentDB.id == incident_id_uuid))
            db_incident = result.scalars().first()

            if not db_incident:
                return None

            reports_result = await session.execute(select(ReportCoreDataDB).where(ReportCoreDataDB.incident_id == incident_id_uuid).order_by(ReportCoreDataDB.timestamp))
            db_reports = reports_result.scalars().all()
            
            p_reports = [await self._report_core_db_to_pydantic(dbr) for dbr in db_reports]
            return await self._incident_db_to_pydantic(db_incident, p_reports)

    async def get_all_incidents(self) -> List[PydanticIncident]:
        async with get_db_session() as session:
            result = await session.execute(select(IncidentDB).order_by(IncidentDB.last_updated_at.desc()))
            db_incidents = result.scalars().all()
            
            p_incidents = []
            for db_inc in db_incidents:
                reports_result = await session.execute(select(ReportCoreDataDB).where(ReportCoreDataDB.incident_id == db_inc.id).order_by(ReportCoreDataDB.timestamp))
                db_reports = reports_result.scalars().all()
                p_reports = [await self._report_core_db_to_pydantic(dbr) for dbr in db_reports]
                p_incidents.append(await self._incident_db_to_pydantic(db_inc, p_reports))
            return p_incidents

    async def get_active_incidents(self) -> List[PydanticIncident]:
        async with get_db_session() as session:
            active_statuses = ["active", "updated", "received", "rcvd", "dispatched", "dsp", "acknowledged", "ack", "enroute", "enr", "onscene", "onscn", "monitoring"]
            # Use 'in_' for multiple values; ensure IncidentDB.status is the correct column name
            result = await session.execute(
                select(IncidentDB).where(IncidentDB.status.in_(active_statuses)).order_by(IncidentDB.last_updated_at.desc()) # type: ignore
            )
            db_incidents = result.scalars().all()
            
            p_incidents = []
            for db_inc in db_incidents:
                reports_result = await session.execute(select(ReportCoreDataDB).where(ReportCoreDataDB.incident_id == db_inc.id).order_by(ReportCoreDataDB.timestamp))
                db_reports = reports_result.scalars().all()
                p_reports = [await self._report_core_db_to_pydantic(dbr) for dbr in db_reports]
                p_incidents.append(await self._incident_db_to_pydantic(db_inc, p_reports))
            return p_incidents

    async def update_incident_status(self, incident_id_str: str, new_status: str) -> bool:
        async with get_db_session() as session:
            try:
                incident_id_uuid = uuid.UUID(incident_id_str)
            except ValueError:
                logger.warning(f"Invalid UUID format for incident_id: {incident_id_str}")
                return False
            
            result = await session.execute(select(IncidentDB).where(IncidentDB.id == incident_id_uuid))
            db_incident = result.scalars().first()

            if db_incident:
                db_incident.status = new_status
                db_incident.last_updated_at = datetime.now(timezone.utc)
                await session.commit()
                logger.info(f"Incident {incident_id_str[:8]} status updated to '{new_status}' in DB.")
                return True
            logger.warning(f"Cannot update status for non-existent incident ID: {incident_id_str}")
            return False

    async def clear_store(self):
        async with get_db_session() as session:
            # Order of deletion matters due to foreign key constraints
            deleted_reports_count = (await session.execute(delete(ReportCoreDataDB))).rowcount # type: ignore
            deleted_incidents_count = (await session.execute(delete(IncidentDB))).rowcount # type: ignore
            await session.commit()
            logger.warning(f"Cleared DB: {deleted_incidents_count} incidents and {deleted_reports_count} reports removed.")


# Global instance
# For FastAPI, dependency injection of sessions is preferred over global store instance for DB operations.
# However, to maintain a similar interface for the agent_core for now, we might adapt.
# Let's make agent_core methods async and pass session or make storage methods callable from endpoints.
# The agent_core itself will now need to be async if it calls these storage methods.
# The agent_core methods that interact with storage must become async and accept/use a db session.

# For now, let's instantiate this, but it will be used by FastAPI endpoints which manage their own sessions.
# The agent_core will need refactoring to work within FastAPI's async/session context.
# A simpler approach for now for `incident_store` being used by non-FastAPI parts or Streamlit (before refactor):
# Create synchronous wrappers or acknowledge that direct use outside FastAPI request lifecycle is problematic with this new DB store.
# Given the UI refactor, direct use from Streamlit will cease. Agent core is called BY FastAPI, so it can use the session.

_incident_store_instance = IncidentStore()

async def get_incident_store() -> IncidentStore:
    # This could be a dependency for FastAPI if we wanted to pass the store instance
    # But individual methods are better for session management.
    return _incident_store_instance


# Quick utility to get a session for scripts or non-FastAPI contexts (use with care)
async def get_standalone_session() -> AsyncSession:
    return AsyncSessionLocal()