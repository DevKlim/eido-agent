import logging
import os
import uuid as uuid_pkg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, String, DateTime, Text, Float, ForeignKey, JSON, Uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from config.settings import settings

logger = logging.getLogger(__name__)

db_url_str = str(settings.database_url) if settings.database_url else ""
final_db_url = db_url_str

# --- Database-Aware Type Definitions ---
if "sqlite" in db_url_str:
    logger.info("Configuring for SQLite database.")
    # SQLite uses standard JSON and Uuid types via SQLAlchemy's generic types.
    JSON_TYPE = JSON
    UUID_TYPE = Uuid
else:
    logger.info("Configuring for PostgreSQL database.")
    # For PostgreSQL, we can use the more optimized, native types.
    from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
    JSON_TYPE = JSONB
    UUID_TYPE = PG_UUID

    # URL Processing for PostgreSQL
    # The 'sslmode' parameter in connection URLs from services like Render
    # can cause a TypeError with asyncpg. We parse the URL, remove 'sslmode',
    # and rebuild it. asyncpg handles SSL automatically based on server
    # requirements or env vars like PGSSLMODE set by Render.
    try:
        parsed_url = urlparse(db_url_str)
        query_params = parse_qs(parsed_url.query)

        if 'sslmode' in query_params:
            logger.info(
                "Removing 'sslmode' from PostgreSQL database URL for asyncpg compatibility.")
            del query_params['sslmode']

            # Rebuild the URL without the 'sslmode' parameter
            url_parts = list(parsed_url)
            url_parts[4] = urlencode(query_params, doseq=True)
            final_db_url = urlunparse(url_parts)
    except Exception as e:
        logger.error(
            f"Failed to parse and rebuild DATABASE_URL. Using original value. Error: {e}")
        # final_db_url remains db_url_str in case of error

if not final_db_url:
    # This will cause a more obvious error than trying to create an engine with an empty string
    raise ValueError(
        "FATAL: DATABASE_URL is not configured. Please set it in your .env file or environment variables."
    )

# Ensure data directory exists for SQLite
if "sqlite" in final_db_url:
    # final_db_url is like "sqlite+aiosqlite:///./data/file.db"
    # We need to extract the path part: "./data/file.db"
    db_path = final_db_url.split("///", 1)[-1]
    db_dir = os.path.dirname(db_path)
    if db_dir:  # Only create if a directory path exists (not just a file name)
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Ensured database directory exists: {db_dir}")

engine = create_async_engine(final_db_url, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

# --- Database Models (Now using generic types) ---


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

    # Define the relationship to ReportCoreDataDB
    # This allows SQLAlchemy to eagerly load reports associated with an incident
    reports = relationship("ReportCoreDataDB", backref="incident", lazy="noload",
                           order_by="ReportCoreDataDB.timestamp", cascade="all, delete-orphan")


async def init_db():
    if not engine:
        logger.critical(
            "Database engine is not initialized. Cannot run init_db().")
        return
    async with engine.begin() as conn:
        logger.info(
            "Initializing database and creating tables if they don't exist...")
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