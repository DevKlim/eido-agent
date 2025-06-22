import logging
import os
import uuid as uuid_pkg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, String, DateTime, Text, Float, ForeignKey, JSON, Uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from config.settings import settings

logger = logging.getLogger(__name__)

db_url_str = str(settings.database_url) if settings.database_url else ""
final_db_url = db_url_str

# --- Database-Aware Type Definitions ---
if "sqlite" in db_url_str:
    logger.info("Configuring for SQLite database.")
    JSON_TYPE = JSON
    # For SQLite, SQLAlchemy's generic Uuid works well.
    # It stores UUIDs as strings but handles the conversion.
    UUID_TYPE = Uuid
else:
    logger.info("Configuring for PostgreSQL database.")
    from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
    JSON_TYPE = JSONB
    # For PostgreSQL, use the native UUID type and specify as_uuid=True
    # to ensure Python's uuid.UUID objects are used.
    UUID_TYPE = PG_UUID(as_uuid=True)

if not final_db_url:
    raise ValueError(
        "FATAL: DATABASE_URL is not configured. Please set it in your .env file or environment variables."
    )

# Ensure data directory exists for SQLite
if "sqlite" in final_db_url:
    db_path = final_db_url.split("///", 1)[-1]
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Ensured database directory exists: {db_dir}")

engine = create_async_engine(final_db_url, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


class ReportCoreDataDB(Base):
    __tablename__ = "reports_core_data"
    id = Column(UUID_TYPE, primary_key=True, default=uuid_pkg.uuid4)
    incident_id = Column(UUID_TYPE, ForeignKey(
        "incidents.id"), nullable=False, index=True)
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
    original_eido_dict = Column(JSON_TYPE, nullable=True)


class IncidentDB(Base):
    __tablename__ = "incidents"
    id = Column(UUID_TYPE, primary_key=True, default=uuid_pkg.uuid4)
    name = Column(String, default="Untitled Incident")
    incident_type = Column(String, nullable=True)
    status = Column(String, default="Active")
    created_at = Column(DateTime(timezone=True), nullable=True)
    last_updated_at = Column(DateTime(timezone=True), nullable=True)
    summary = Column(Text, default="Summary not yet generated.")
    recommended_actions = Column(JSON_TYPE, default=list)
    locations_coords = Column(JSON_TYPE, default=list)
    addresses = Column(JSON_TYPE, default=list)
    zip_codes = Column(JSON_TYPE, default=list)
    trend_data = Column(JSON_TYPE, default=dict)
    reports = relationship("ReportCoreDataDB", backref="incident", lazy="noload",
                           order_by="ReportCoreDataDB.timestamp", cascade="all, delete-orphan")


async def init_db():
    """
    Creates tables if they do not exist. In a development environment,
    you might uncomment the drop_all line to reset the DB on each start.
    For production, drop_all should always be commented out to ensure data persistence.
    """
    if not engine:
        logger.critical(
            "Database engine is not initialized. Cannot run init_db().")
        return
    async with engine.begin() as conn:
        # The following line is DESTRUCTIVE and should be commented out for production
        # to ensure data persistence across restarts.
        # logger.warning("Dropping all existing database tables...")
        # await conn.run_sync(Base.metadata.drop_all)
        # logger.info("All tables dropped successfully.")
        
        logger.info("Ensuring all tables exist in the database...")
        # create_all with checkfirst=True (default) is safe and non-destructive.
        # It will only create tables that do not already exist.
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database table check/creation complete.")


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