# config/settings.py
import os
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional, Any

# --- Import Streamlit ---
# Use a try-except block in case streamlit is not installed or running outside streamlit context
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None # Assign None if streamlit is not available
    STREAMLIT_AVAILABLE = False

# Load .env file variables into environment variables (for local development)
# This should run BEFORE BaseSettings reads the environment
# It's safe to call even if .env doesn't exist
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Construct path relative to this file
load_dotenv(dotenv_path=dotenv_path)

# --- Logger Setup ---
# Basic setup here, might be further configured by the application entry point
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.debug(f"Attempting to load settings. Streamlit available: {STREAMLIT_AVAILABLE}")
if not STREAMLIT_AVAILABLE:
    logger.warning("Streamlit library not found or import failed. Will only use environment variables for settings.")


# --- Function to safely get config values ---
def get_config_value(key: str, env_var: str, default: Optional[Any] = None) -> Optional[Any]:
    """
    Tries to get a value from st.secrets first (if streamlit is available),
    then from environment variables, then returns default.
    Handles potential errors if st.secrets is accessed improperly.
    """
    value = None
    secrets_checked = False

    # Try st.secrets only if Streamlit is available and running
    if STREAMLIT_AVAILABLE and hasattr(st, 'secrets'):
        try:
            # Check if the key exists directly in st.secrets
            if key in st.secrets:
                value = st.secrets[key]
                logger.debug(f"Retrieved '{key}' from st.secrets.")
            # Example: Check for nested structures if you organize secrets.toml that way
            # elif "group_name" in st.secrets and key in st.secrets["group_name"]:
            #     value = st.secrets["group_name"][key]
            #     logger.debug(f"Retrieved '{key}' from st.secrets nested under 'group_name'.")
            secrets_checked = True
        except Exception as e:
            # This might happen if accessed outside a running Streamlit app context
            # even if the library is installed.
            logger.warning(f"Could not access st.secrets for key '{key}'. Error: {e}. Falling back to environment variables.")
            secrets_checked = False # Ensure fallback happens

    # Fallback to environment variable if not found in st.secrets or st.secrets not available/accessible
    if value is None:
        env_value = os.getenv(env_var)
        if env_value is not None:
            value = env_value
            if secrets_checked:
                 logger.debug(f"Key '{key}' not found in st.secrets, using environment variable '{env_var}'.")
            else:
                 logger.debug(f"Using environment variable '{env_var}' for key '{key}'.")
        else:
            # Use default if environment variable is also not set
            value = default
            logger.debug(f"Using default value for key '{key}'.")

    # Return the found value or the default
    return value


# --- Settings Definition using Pydantic BaseSettings ---
class Settings(BaseSettings):
    """
    Defines application settings, loading values prioritizing st.secrets,
    then environment variables, then defaults.
    """
    # --- Required ---
    # Note: Defaults provided here are fallback if neither st.secrets nor ENV var is set.
    geocoding_user_agent: str = get_config_value(
        key="GEOCODING_USER_AGENT",
        env_var="GEOCODING_USER_AGENT",
        default="EidoAgent/1.0 (contact@example.com)" # Provide a generic default
    )

    # --- LLM Configuration ---
    llm_provider: str = get_config_value(
        key="LLM_PROVIDER",
        env_var="LLM_PROVIDER",
        default="none" # Default to 'none' if not specified
    ).lower() # Ensure lowercase for comparisons

    # Google Specific
    google_api_key: Optional[str] = get_config_value(
        key="GOOGLE_API_KEY",
        env_var="GOOGLE_API_KEY",
        default=None
    )
    google_model_name: str = get_config_value(
        key="GOOGLE_MODEL_NAME",
        env_var="GOOGLE_MODEL_NAME",
        default="gemini-1.5-flash-latest"
    )

    # OpenRouter Specific (Define attributes even if not primary provider)
    # Use generic names that can hold keys/models for different providers based on llm_provider
    llm_api_key: Optional[str] = get_config_value(
        key="LLM_API_KEY", # Could be GOOGLE_API_KEY or OPENROUTER_API_KEY etc. in secrets/env
        env_var="LLM_API_KEY",
        default=None
    )
    llm_model_name: Optional[str] = get_config_value(
        key="LLM_MODEL_NAME", # Could be GOOGLE_MODEL_NAME or OPENROUTER_MODEL_NAME etc.
        env_var="LLM_MODEL_NAME",
        default=None
    )
    llm_api_base_url: Optional[str] = get_config_value(
        key="LLM_API_BASE_URL", # e.g. https://openrouter.ai/api/v1
        env_var="LLM_API_BASE_URL",
        default=None
    )

    # --- Embedding Model ---
    embedding_model_name: str = get_config_value(
        key="EMBEDDING_MODEL_NAME",
        env_var="EMBEDDING_MODEL_NAME",
        default="all-MiniLM-L6-v2"
    )

    # --- Incident Matching Parameters ---
    # Ensure type conversion after getting the value
    similarity_threshold: float = float(get_config_value(
        key="SIMILARITY_THRESHOLD",
        env_var="SIMILARITY_THRESHOLD",
        default="0.70" # Keep default as string for get_config_value
    ))
    time_window_minutes: int = int(get_config_value(
        key="TIME_WINDOW_MINUTES",
        env_var="TIME_WINDOW_MINUTES",
        default="60" # Keep default as string
    ))
    distance_threshold_km: float = float(get_config_value(
        key="DISTANCE_THRESHOLD_KM",
        env_var="DISTANCE_THRESHOLD_KM",
        default="1.0" # Keep default as string
    ))

    # --- API/UI Server Ports (Usually set by platform/run script via ENV) ---
    # Keep reading these directly from environment variables as they are less likely
    # to be in st.secrets and often controlled by the deployment environment.
    api_port: int = int(os.getenv("API_PORT", "8000"))
    streamlit_server_port: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

    # --- Logging ---
    # Log level is used by the logger setup itself, read directly or via get_config_value
    log_level: str = get_config_value(
        key="LOG_LEVEL",
        env_var="LOG_LEVEL",
        default="INFO"
    ).upper() # Ensure uppercase for logging module

    # --- Alert Parsing Prompt ---
    alert_parsing_prompt_path: Optional[str] = get_config_value(
        key="ALERT_PARSING_PROMPT_PATH",
        env_var="ALERT_PARSING_PROMPT_PATH",
        default=None
    )

    # --- Pydantic V2 Configuration ---
    # Use model_config instead of class Config for Pydantic V2
    model_config = {
        "extra": "ignore" # Ignore extra fields from env/secrets if any
    }

# --- Instantiate Settings ---
# Wrap in try-except to catch validation errors from Pydantic
try:
    settings = Settings()
    # Log loaded settings (be careful not to log secrets like API keys!)
    logger.info("Settings loaded successfully.")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Log Level: {settings.log_level}")
    logger.debug(f"Geocoding Agent: {settings.geocoding_user_agent}")
    logger.debug(f"Embedding Model: {settings.embedding_model_name}")
    logger.debug(f"Similarity Threshold: {settings.similarity_threshold}")
    # Add logging for other non-sensitive settings if needed for debugging

    # --- Post-load adjustments if necessary ---
    # Example: If using Google, ensure llm_api_key and llm_model_name reflect Google settings
    # This assumes you store GOOGLE_API_KEY and GOOGLE_MODEL_NAME in secrets/env
    if settings.llm_provider == 'google':
        settings.llm_api_key = settings.google_api_key
        settings.llm_model_name = settings.google_model_name
        logger.debug("Adjusted generic LLM settings for Google provider.")
    # Add similar blocks for 'openrouter' or other providers if needed

except Exception as e:
    logger.critical(f"CRITICAL ERROR: Failed to initialize settings: {e}", exc_info=True)
    # Depending on the application structure, you might want to exit or raise
    raise SystemExit(f"Failed to initialize settings: {e}") from e

# --- Example Usage (in other modules) ---
# from config.settings import settings
#
# api_key = settings.llm_api_key # Will be None if not set via secrets or ENV
# level = settings.log_level