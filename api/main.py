# api/main.py
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Optional: For handling CORS if UI is served separately

# Import settings and the API router
from config.settings import settings
from api.endpoints import router as api_router # Import the router instance

# --- Logging Configuration ---
# Configure logging based on settings before creating app instance
log_level = settings.log_level.upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"API log level set to: {log_level}")

# --- FastAPI Application Instance ---
app = FastAPI(
    title="EIDO Sentinel API",
    description="API for ingesting EIDO reports and managing emergency incidents.",
    version="0.2.0", # Increment version
    # Optional: Add contact info, license info etc. for OpenAPI docs
    # contact={"name": "Support Team", "email": "support@example.com"},
    # license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
)

# --- Middleware ---
# Optional: Add CORS middleware if your Streamlit UI is on a different origin (port)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost", f"http://localhost:{settings.streamlit_server_port}"], # Origins allowed to make requests
#     allow_credentials=True,
#     allow_methods=["*"], # Allows all methods (GET, POST, etc.)
#     allow_headers=["*"], # Allows all headers
# )
# logger.info("CORS middleware added (if uncommented).")


# --- Include API Routers ---
app.include_router(api_router, prefix="/api/v1") # Prefix all API routes
logger.info("API router included with prefix /api/v1")


# --- Root Endpoint ---
@app.get("/", summary="API Root", description="Basic health check endpoint for the API.", include_in_schema=False) # Hide from docs
async def read_root():
    return {"message": "EIDO Sentinel API is running.", "version": app.version}

# --- Main execution block to run with Uvicorn ---
# This allows running the API directly using `python api/main.py` (though `uvicorn` command is preferred)
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "api.main:app", # Point to the FastAPI app instance
        host=settings.api_host,
        port=settings.api_port,
        reload=True, # Enable auto-reload for development (consider disabling in production)
        log_level=log_level.lower() # Pass log level to uvicorn
    )