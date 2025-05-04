# config/settings.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from typing import List, Optional, Literal, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Define allowed LLM providers
LlmProvider = Literal['google', 'openrouter', 'local', 'none']

# --- Define allowed Google models based on your request ---
# Note: Check Google's official documentation for currently available/recommended models.
# These names seem plausible based on common patterns, but verify them.
# Removed gemini-2.0-flash and gemini-2.0-flash-lite as they might not be standard names.
# Added gemini-1.0-pro as a stable option.
GOOGLE_MODEL_OPTIONS = [
    "gemini-2.0-flash", # Recommended Flash model
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",   # Recommended Pro model
    "gemini-2.0-flash-lite",          # Stable older Pro model
    # Add specific preview names if absolutely needed, but prefer stable ones
    # "gemini-pro-1.5-pro-preview-0514", # Example of a preview name pattern
]
# --- Set a default that IS in the options list ---
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash" # Ensure this exists in GOOGLE_MODEL_OPTIONS

class Settings(BaseSettings):
    """Application Settings"""

    # --- General ---
    app_name: str = "EIDO Sentinel"
    log_level: str = Field("INFO", validation_alias='LOG_LEVEL')

    # --- LLM Configuration ---
    llm_provider: LlmProvider = Field("google", validation_alias='LLM_PROVIDER') # Default to google

    # --- Google Specific ---
    google_api_key: Optional[str] = Field(None, validation_alias='GOOGLE_API_KEY')
    # Use the defined default model
    google_model_name: str = Field(DEFAULT_GEMINI_MODEL, validation_alias='GOOGLE_MODEL_NAME')
    # Store the allowed options for the UI
    google_model_options: List[str] = Field(default_factory=lambda: GOOGLE_MODEL_OPTIONS)

    # --- OpenRouter Specific ---
    openrouter_api_key: Optional[str] = Field(None, validation_alias='OPENROUTER_API_KEY')
    openrouter_model_name: Optional[str] = Field(None, validation_alias='OPENROUTER_MODEL_NAME') # e.g., "openai/gpt-4o-mini"
    openrouter_api_base_url: str = Field("https://openrouter.ai/api/v1", validation_alias='OPENROUTER_API_BASE_URL')

    # --- Local LLM Specific (Ollama/LMStudio via OpenAI client) ---
    local_llm_api_key: Optional[str] = Field("EMPTY", validation_alias='LOCAL_LLM_API_KEY') # Often not needed or use placeholder like 'ollama'
    local_llm_model_name: Optional[str] = Field(None, validation_alias='LOCAL_LLM_MODEL_NAME') # Must be set by user, e.g., "llama3:latest"
    local_llm_api_base_url: Optional[str] = Field(None, validation_alias='LOCAL_LLM_API_BASE_URL') # e.g., "http://localhost:11434/v1" or "http://localhost:1234/v1"

    # --- Embedding Configuration ---
    embedding_model_name: str = Field("all-MiniLM-L6-v2", validation_alias='EMBEDDING_MODEL_NAME')

    # --- Geocoding Configuration ---
    geocoding_user_agent: str = Field("EidoSentinelApp/0.6 (Contact: your_email@example.com)", validation_alias='GEOCODING_USER_AGENT')

    # --- Matching Configuration ---
    similarity_threshold: float = Field(0.70, validation_alias='SIMILARITY_THRESHOLD')
    time_window_minutes: int = Field(60, validation_alias='TIME_WINDOW_MINUTES')
    distance_threshold_km: float = Field(1.0, validation_alias='DISTANCE_THRESHOLD_KM') # Adjusted based on previous code

    # --- API Configuration ---
    api_host: str = Field("127.0.0.1", validation_alias='API_HOST') # Changed from 0.0.0.0 for clarity unless binding to all is needed
    api_port: int = Field(8000, validation_alias='API_PORT')

    # --- UI Configuration ---
    streamlit_server_port: int = Field(8501, validation_alias='STREAMLIT_SERVER_PORT')

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=('.env', 'project/.env'), # Look for .env in current dir first
        extra='ignore'
    )

    # --- Validation ---
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        upper_v = v.upper()
        if upper_v not in allowed_levels:
            raise ValueError(f"Invalid log_level '{v}'. Must be one of {allowed_levels}")
        return upper_v # Return the validated upper-case value

    @field_validator('llm_provider')
    @classmethod
    def validate_llm_provider(cls, v):
        # The Literal type handles validation, but an explicit check can be clearer
        if v not in ['google', 'openrouter', 'local', 'none']:
            raise ValueError("Invalid llm_provider. Must be 'google', 'openrouter', 'local', or 'none'.")
        return v

    # Ensure the selected google_model_name is actually in the allowed options list
    @field_validator('google_model_name')
    @classmethod
    def validate_google_model_name(cls, v, values):
        # Need access to the options list defined earlier
        # Pydantic v2: `values` is deprecated in field_validator, use ValidationInfo
        # Let's simplify - we'll check this during UI interaction instead or assume the default is valid.
        # Or better, check against the class variable default list:
        if v not in GOOGLE_MODEL_OPTIONS:
             logger.warning(f"Configured google_model_name '{v}' is not in the predefined list GOOGLE_MODEL_OPTIONS. Using default: {DEFAULT_GEMINI_MODEL}")
             return DEFAULT_GEMINI_MODEL # Fallback to default if invalid name provided
        return v


    @model_validator(mode='before')
    @classmethod
    def check_required_llm_fields(cls, data: Any) -> Any:
        """ Check if required fields are present based on the LLM provider. """
        if not isinstance(data, dict):
            logger.warning(f"Model validator received non-dict data: {type(data)}. Skipping LLM field checks.")
            return data

        provider = data.get('LLM_PROVIDER') # Check using the env var name

        if provider == 'google':
            if not data.get('GOOGLE_API_KEY'):
                logger.warning("LLM Provider is 'google' but GOOGLE_API_KEY is not set in environment.")
        elif provider == 'openrouter':
            if not data.get('OPENROUTER_API_KEY'):
                logger.warning("LLM Provider is 'openrouter' but OPENROUTER_API_KEY is not set in environment.")
            if not data.get('OPENROUTER_MODEL_NAME'):
                logger.warning("LLM Provider is 'openrouter' but OPENROUTER_MODEL_NAME is not set in environment.")
        elif provider == 'local':
            if not data.get('LOCAL_LLM_MODEL_NAME'):
                logger.warning("LLM Provider is 'local' but LOCAL_LLM_MODEL_NAME is not set in environment.")
            if not data.get('LOCAL_LLM_API_BASE_URL'):
                logger.warning("LLM Provider is 'local' but LOCAL_LLM_API_BASE_URL is not set in environment.")

        geo_agent = data.get('GEOCODING_USER_AGENT')
        if not geo_agent or 'your_email@example.com' in geo_agent or 'example@example.com' in geo_agent: # Check both old/new examples
             logger.warning("GEOCODING_USER_AGENT is not set or uses the default example. Please configure it with your application name and contact information as per Nominatim's usage policy.")

        return data

# --- Create a singleton instance ---
# Remove the get_settings() function and create instance directly
try:
    settings = Settings()
    # Log loaded settings (be careful with keys)
    logger.info("Settings loaded successfully.")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Log Level: {settings.log_level}")
    if settings.llm_provider == 'google':
        logger.info(f"Google Model: {settings.google_model_name}")
    elif settings.llm_provider == 'openrouter':
        logger.info(f"OpenRouter Model: {settings.openrouter_model_name}")
    elif settings.llm_provider == 'local':
        logger.info(f"Local LLM Model: {settings.local_llm_model_name}")
        logger.info(f"Local LLM URL: {settings.local_llm_api_base_url}")

except Exception as e:
     logger.critical(f"CRITICAL ERROR: Failed to load settings: {e}", exc_info=True)
     raise SystemExit(f"Failed to initialize settings: {e}") from e