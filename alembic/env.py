import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ==============================================================================
# 1. PATH SETUP
#
# This allows Alembic to find your application's models.
# It adds the project's root directory to the Python path.
# ==============================================================================
# Resolve the project root directory (assuming this file is in 'alembic/')
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


# ==============================================================================
# 2. ALEMBIC CONFIGURATION
#
# This is the standard Alembic config object.
# ==============================================================================
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ==============================================================================
# 3. MODEL METADATA
#
# Point Alembic to your SQLAlchemy models' Base.metadata.
# This is crucial for autogenerate to work.
#
# !!! IMPORTANT !!!
# Make sure the import path below is correct for your project structure.
# Based on your error logs, it's likely in 'app/models/incident.py'.
# ==============================================================================
try:
    from app.models.incident import Base
    target_metadata = Base.metadata
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error: Could not import the SQLAlchemy Base model. Please check the import path in alembic/env.py.")
    print(f"Tried to import from 'app.models.incident'. Original error: {e}")
    sys.exit(1)


# ==============================================================================
# 4. DATABASE URL CONFIGURATION (THE FIX IS HERE)
#
# This section reads the DATABASE_URL from the environment (your Fly.io secret),
# and modifies it for synchronous use by Alembic.
# ==============================================================================
def get_url():
    """
    Gets the database URL from the environment and makes it sync-compatible.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "DATABASE_URL environment variable is not set. "
            "Please set it in your Fly.io secrets."
        )

    # Alembic is a synchronous tool. If the URL is for an async driver like
    # 'asyncpg', we must convert it to a standard synchronous one.
    # SQLAlchemy will then use the 'psycopg2-binary' driver you installed.
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)

    return url

# Set the final, corrected URL in the Alembic configuration so that
# SQLAlchemy can use it to connect to the database.
config.set_main_option('sqlalchemy.url', get_url())


# ==============================================================================
# 5. ALEMBIC MIGRATION FUNCTIONS
#
# These are the standard functions Alembic uses to run migrations.
# ==============================================================================
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Create an engine from the config, which already has the corrected URL
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


# This is the main entry point for Alembic.
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()