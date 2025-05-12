import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, DateTime, Text, Float, ForeignKey, JSON, dialects
from sqlalchemy.dialects.postgresql import UUID as PG_UUID # For PostgreSQL UUID type
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from config.settings import settings

logger = logging.getLogger(__name__)

DATABASE_URL = str(settings.database_url)

engine = create_async_engine(DATABASE_URL, echo=False) # Set echo=True for SQL logging

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

# --- Database Models ---
class IncidentDB(Base):
    __tablename__ = "incidents"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_type = Column(String, nullable=True)
    status = Column(String, default="Active")
    created_at = Column(DateTime(timezone=True), nullable=True)
    last_updated_at = Column(DateTime(timezone=True), nullable=True)
    summary = Column(Text, default="Summary not yet generated.")
    
    # Using JSONB for PostgreSQL for these list/dict fields
    recommended_actions = Column(dialects.postgresql.JSONB, default=list)
    locations_coords = Column(dialects.postgresql.JSONB, default=list) # List of [lat, lon]
    addresses = Column(dialects.postgresql.JSONB, default=list) # List of strings
    zip_codes = Column(dialects.postgresql.JSONB, default=list) # List of strings
    trend_data = Column(dialects.postgresql.JSONB, default=dict)

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


async def init_db():
    """Initializes the database and creates tables if they don't exist."""
    async with engine.begin() as conn:
        logger.info("Initializing database and creating tables if they don't exist...")
        # await conn.run_sync(Base.metadata.drop_all)  # Use with caution: drops all tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables checked/created.")

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional scope around a series of operations."""
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