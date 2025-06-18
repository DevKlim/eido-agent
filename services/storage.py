import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload # Import selectinload
import json 
from sqlalchemy import delete

from data_models.schemas import Incident as PydanticIncident, ReportCoreData as PydanticReportCoreData
from services.database import get_db_session, IncidentDB, ReportCoreDataDB, AsyncSessionLocal

logger = logging.getLogger(__name__)

class IncidentStore:
    def __init__(self):
        logger.info("Database-backed IncidentStore initialized.")

    async def _pydantic_to_incident_db(self, p_incident: PydanticIncident) -> IncidentDB:
        return IncidentDB(
            id=uuid.UUID(p_incident.incident_id) if isinstance(p_incident.incident_id, str) else p_incident.incident_id,
            name=p_incident.name,
            incident_type=p_incident.incident_type,
            status=p_incident.status,
            created_at=p_incident.created_at,
            last_updated_at=p_incident.last_updated_at,
            summary=p_incident.summary,
            recommended_actions=p_incident.recommended_actions, 
            locations_coords=[list(loc) if isinstance(loc, tuple) else loc for loc in p_incident.locations], 
            addresses=p_incident.addresses,
            zip_codes=p_incident.zip_codes,
            trend_data=p_incident.trend_data
        )

    async def _incident_db_to_pydantic(self, db_incident: IncidentDB, reports_core_data: List[PydanticReportCoreData]) -> PydanticIncident:
        # If reports_core_data is not passed, it means it was eagerly loaded via db_incident.reports
        if not reports_core_data and db_incident.reports: # Check if reports were eagerly loaded
             reports_core_data = [await self._report_core_db_to_pydantic(dbr) for dbr in db_incident.reports]

        return PydanticIncident(
            incident_id=str(db_incident.id),
            name=db_incident.name,
            incident_type=db_incident.incident_type,
            status=db_incident.status,
            created_at=db_incident.created_at,
            last_updated_at=db_incident.last_updated_at,
            summary=db_incident.summary,
            recommended_actions=db_incident.recommended_actions if isinstance(db_incident.recommended_actions, list) else [],
            locations=[tuple(loc) if isinstance(loc, list) else loc for loc in (db_incident.locations_coords or [])], 
            addresses=db_incident.addresses if isinstance(db_incident.addresses, list) else [],
            zip_codes=db_incident.zip_codes if isinstance(db_incident.zip_codes, list) else [],
            trend_data=db_incident.trend_data if isinstance(db_incident.trend_data, dict) else {},
            reports_core_data=reports_core_data
        )

    async def _pydantic_to_report_core_db(self, p_report: PydanticReportCoreData, incident_id_uuid: uuid.UUID) -> ReportCoreDataDB:
        coords_lat, coords_lon = (p_report.coordinates[0], p_report.coordinates[1]) if p_report.coordinates else (None, None)
        
        original_eido_dict_serializable = None
        if p_report.original_eido_dict:
            try:
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
            original_eido_dict=original_eido_dict_serializable 
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
            
            result = await session.execute(
                select(IncidentDB)
                .options(selectinload(IncidentDB.reports)) # Eagerly load reports for potential update logic
                .where(IncidentDB.id == incident_id_uuid)
            )
            db_incident = result.scalars().first()

            if db_incident: 
                db_incident.name = p_incident.name
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
            else: 
                db_incident = await self._pydantic_to_incident_db(p_incident)
                session.add(db_incident)
                logger.debug(f"Saving new Incident {p_incident.incident_id[:8]} to DB.")

            # Efficiently sync reports:
            # 1. Get current report IDs from DB for this incident
            # (This step can be skipped if we always delete and re-add, but is good for more fine-grained updates)
            # current_db_report_ids = {report.id for report in db_incident.reports} if db_incident and db_incident.reports else set()
            
            # For simplicity and ensuring Pydantic model is source of truth: delete existing and re-add
            # This is what was done before and is fine if performance is not an issue for report syncing.
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

            result = await session.execute(
                select(IncidentDB)
                .options(selectinload(IncidentDB.reports)) # Eagerly load reports
                .where(IncidentDB.id == incident_id_uuid)
            )
            db_incident = result.scalars().first()

            if not db_incident:
                return None
            
            # Reports are already loaded in db_incident.reports due to selectinload
            p_reports = [await self._report_core_db_to_pydantic(dbr) for dbr in db_incident.reports]
            return await self._incident_db_to_pydantic(db_incident, p_reports)

    async def get_all_incidents(self) -> List[PydanticIncident]:
        async with get_db_session() as session:
            result = await session.execute(
                select(IncidentDB)
                .options(selectinload(IncidentDB.reports)) # Eagerly load reports
                .order_by(IncidentDB.last_updated_at.desc())
            )
            db_incidents = result.scalars().unique().all() # Use .unique() when using eager loading strategies like selectinload
            
            p_incidents = []
            for db_inc in db_incidents:
                # Reports are already loaded in db_inc.reports
                p_reports = [await self._report_core_db_to_pydantic(dbr) for dbr in db_inc.reports]
                p_incidents.append(await self._incident_db_to_pydantic(db_inc, p_reports))
            return p_incidents

    async def get_active_incidents(self) -> List[PydanticIncident]:
        async with get_db_session() as session:
            active_statuses = ["active", "updated", "received", "rcvd", "dispatched", "dsp", "acknowledged", "ack", "enroute", "enr", "onscene", "onscn", "monitoring"]
            result = await session.execute(
                select(IncidentDB)
                .options(selectinload(IncidentDB.reports)) # Eagerly load reports
                .where(IncidentDB.status.in_(active_statuses)) # type: ignore
                .order_by(IncidentDB.last_updated_at.desc())
            )
            db_incidents = result.scalars().unique().all() # Use .unique()
            
            p_incidents = []
            for db_inc in db_incidents:
                # Reports are already loaded in db_inc.reports
                p_reports = [await self._report_core_db_to_pydantic(dbr) for dbr in db_inc.reports]
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
            deleted_reports_count = (await session.execute(delete(ReportCoreDataDB))).rowcount # type: ignore
            deleted_incidents_count = (await session.execute(delete(IncidentDB))).rowcount # type: ignore
            await session.commit()
            logger.warning(f"Cleared DB: {deleted_incidents_count} incidents and {deleted_reports_count} reports removed.")

_incident_store_instance = IncidentStore()

async def get_incident_store() -> IncidentStore:
    return _incident_store_instance

async def get_standalone_session() -> AsyncSession:
    return AsyncSessionLocal()