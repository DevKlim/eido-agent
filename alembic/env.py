import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- START OF THE IMPORTANT PART ---

# Get the database URL from the environment variable
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL environment variable not set")

# Alembic is synchronous, so we need to use a synchronous dialect.
# If the URL is for asyncpg, replace it with the standard postgresql dialect.
# SQLAlchemy will then use the installed psycopg2 driver.
if database_url.startswith("postgresql+asyncpg://"):
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

# Set the modified URL in the Alembic config for SQLAlchemy to use.
config.set_main_option('sqlalchemy.url', database_url)

# --- END OF THE IMPORTANT PART ---

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
from app.models.incident import Base  # <-- Make sure this points to your SQLAlchemy Base
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def run_migrations_offline() -> None:
    # ... (rest of the file is likely fine) ...

def run_migrations_online() -> None:
    # ... (rest of the file is likely fine) ...