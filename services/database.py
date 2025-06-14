import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, String, DateTime, Text, Float, ForeignKey, JSON, dialects
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from config.settings import settings

logger = logging.getLogger(__name__)

db_url_str = str(settings.database_url) if settings.database_url else ""
final_db_url = db_url_str

# --- Database URL Processing for Deployment Compatibility ---
# The 'sslmode' parameter in connection URLs from services like Render
# can cause a TypeError with asyncpg. We parse the URL, remove 'sslmode',
# and rebuild it. asyncpg handles SSL automatically based on server
# requirements or env vars like PGSSLMODE set by Render.
if db_url_str.startswith("postgresql+asyncpg://"):
    try:
        parsed_url = urlparse(db_url_str)
        query_params = parse_qs(parsed_url.query)

        if 'sslmode' in query_params:
            logger.info("Removing 'sslmode' from database URL for asyncpg compatibility.")
            del query_params['sslmode']
            
            # Rebuild the URL without the 'sslmode' parameter
            url_parts = list(parsed_url)
            url_parts[4] = urlencode(query_params, doseq=True)
            final_db_url = urlunparse(url_parts)
    except Exception as e:
        logger.error(f"Failed to parse and rebuild DATABASE_URL. Using original value. Error: {e}")
        final_db_url = db_url_str

if not final_db_url:
    # This will cause a more obvious error than trying to create an engine with an empty string
    raise ValueError("FATAL: DATABASE_URL is not configured in environment settings.")

engine = create_async_engine(final_db_url, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

# --- Database Models ---
class ReportCoreDataDB(Base):
    __tablename__ = "reports_core_data"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(PG_UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    
    external_incident_id = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    incident_type = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    location_address = Column(String, nullable=True)
    coordinates_lat = Column(Float, nullable=True)
    coordinates_lon = Column(Float, nullable=True)
    zip_code = Column(String, nullable=True)
    source = Column(String, nullable=True)
    original_document_id = Column(String, nullable=True)
    original_eido_dict = Column(dialects.postgresql.JSONB, nullable=True)

class IncidentDB(Base):
    __tablename__ = "incidents"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_type = Column(String, nullable=True)
    status = Column(String, default="Active")
    created_at = Column(DateTime(timezone=True), nullable=True)
    last_updated_at = Column(DateTime(timezone=True), nullable=True)
    summary = Column(Text, default="Summary not yet generated.")
    
    recommended_actions = Column(dialects.postgresql.JSONB, default=list)
    locations_coords = Column(dialects.postgresql.JSONB, default=list) 
    addresses = Column(dialects.postgresql.JSONB, default=list) 
    zip_codes = Column(dialects.postgresql.JSONB, default=list) 
    trend_data = Column(dialects.postgresql.JSONB, default=dict)

    # Define the relationship to ReportCoreDataDB
    # This allows SQLAlchemy to eagerly load reports associated with an incident
    reports = relationship("ReportCoreDataDB", backref="incident", lazy="noload", order_by="ReportCoreDataDB.timestamp")


async def init_db():
    if not engine:
        logger.critical("Database engine is not initialized. Cannot run init_db().")
        return
    async with engine.begin() as conn:
        logger.info("Initializing database and creating tables if they don't exist...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables checked/created.")

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async_session = AsyncSessionLocal()
    try:
        yield async_session
        await async_session.commit()
        logger.debug("DB session commit successful.")
    except Exception as e:
        await async_session.rollback()
        logger.error(f"DB session rollback due to error: {e}", exc_info=True)
        raise
    finally:
        await async_session.close()
        logger.debug("DB session closed.")