import asyncio
import logging
import os
import sys

# Add project root to path to allow imports from other packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the init_db function after setting the path
from services.database import init_db
from config.settings import settings

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger("DBInitScript")

async def main():
    """
    Main function to run the database initialization process.
    """
    logger.info("Starting database initialization...")
    try:
        await init_db()
        logger.info("Database initialization completed successfully.")
        sys.exit(0) # Explicitly exit with success code
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}", exc_info=True)
        # Exit with a non-zero code to signal failure to the deployment process
        sys.exit(1)

if __name__ == "__main__":
    # This script will be run by the release_command on Fly.io
    asyncio.run(main())