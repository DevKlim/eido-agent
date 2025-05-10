import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from typing import List, Optional, Literal, Dict, Any
import logging

logger = logging.getLogger(__name__)

LlmProvider = Literal['google', 'openrouter', 'local', 'none']

# Updated Google Model Options based on user request from prompt
GOOGLE_MODEL_OPTIONS = [
    "gemini-2.0-flash",         # Recommended Flash model by user
    "gemini-2.5-flash-preview-04-17", # User provided
    "gemini-2.5-pro-preview-05-06",   # Recommended Pro model by user
    "gemini-2.0-flash-lite",          # User provided (might be same as 2.0-flash or older)
    "gemini-1.5-flash-latest",        # Good general purpose flash from original .env.example
    "gemini-1.5-pro-latest",          # Good general purpose pro
    "gemini-1.0-pro",                 # Stable older Pro model
]
# Ensure default is in the list
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
if DEFAULT_GEMINI_MODEL not in GOOGLE_MODEL_OPTIONS:
    DEFAULT_GEMINI_MODEL = GOOGLE_MODEL_OPTIONS[0] if GOOGLE_MODEL_OPTIONS else "gemini-1.5-flash-latest"


class Settings(BaseSettings):
    app_name: str = "EIDO Sentinel"
    log_level: str = Field("INFO", validation_alias='LOG_LEVEL')

    llm_provider: LlmProvider = Field("google", validation_alias='LLM_PROVIDER')

    google_api_key: Optional[str] = Field(None, validation_alias='GOOGLE_API_KEY')
    google_model_name: str = Field(DEFAULT_GEMINI_MODEL, validation_alias='GOOGLE_MODEL_NAME')
    google_model_options: List[str] = Field(default_factory=lambda: GOOGLE_MODEL_OPTIONS)

    openrouter_api_key: Optional[str] = Field(None, validation_alias='OPENROUTER_API_KEY')
    openrouter_model_name: Optional[str] = Field("openai/gpt-4o-mini", validation_alias='OPENROUTER_MODEL_NAME')
    openrouter_api_base_url: str = Field("https://openrouter.ai/api/v1", validation_alias='OPENROUTER_API_BASE_URL')

    local_llm_api_key: Optional[str] = Field("EMPTY", validation_alias='LOCAL_LLM_API_KEY')
    local_llm_model_name: Optional[str] = Field("llama3:latest", validation_alias='LOCAL_LLM_MODEL_NAME')
    local_llm_api_base_url: Optional[str] = Field("http://localhost:11434/v1", validation_alias='LOCAL_LLM_API_BASE_URL')

    embedding_model_name: str = Field("all-MiniLM-L6-v2", validation_alias='EMBEDDING_MODEL_NAME')
    
    # Using the updated user agent from the prompt
    geocoding_user_agent: str = Field("EidoSentinelApp/0.8 (contact: your_email@example.com)", validation_alias='GEOCODING_USER_AGENT')

    similarity_threshold: float = Field(0.70, validation_alias='SIMILARITY_THRESHOLD')
    time_window_minutes: int = Field(60, validation_alias='TIME_WINDOW_MINUTES')
    distance_threshold_km: float = Field(1.0, validation_alias='DISTANCE_THRESHOLD_KM')

    api_host: str = Field("127.0.0.1", validation_alias='API_HOST')
    api_port: int = Field(8000, validation_alias='API_PORT')

    streamlit_server_port: int = Field(8501, validation_alias='STREAMLIT_SERVER_PORT')

    model_config = SettingsConfigDict(
        env_file=('.env', '.env.production', '.env.development'), # Standard .env loading
        extra='ignore',
        env_file_encoding='utf-8'
    )

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed: raise ValueError(f"Invalid log_level '{v}'. Must be one of {allowed}")
        return v.upper()

    @field_validator('google_model_name')
    @classmethod
    def validate_google_model_name(cls, v: str, info: Any): # Use info for Pydantic v2
        # Access GOOGLE_MODEL_OPTIONS from the class level or info.data if pre-populated
        # For simplicity, using the class-level const
        if v not in GOOGLE_MODEL_OPTIONS:
             logger.warning(f"Configured google_model_name '{v}' is not in GOOGLE_MODEL_OPTIONS. Using default: {DEFAULT_GEMINI_MODEL}")
             return DEFAULT_GEMINI_MODEL
        return v

    @model_validator(mode='before') # mode='before' to access raw env var names
    @classmethod
    def check_llm_dependencies(cls, data: Any) -> Any:
        if not isinstance(data, dict): return data # Should be a dict from .env

        provider = data.get('LLM_PROVIDER', 'none') # Get from .env name
        
        if provider == 'google' and not data.get('GOOGLE_API_KEY'):
            logger.warning("LLM_PROVIDER is 'google' but GOOGLE_API_KEY is missing.")
        elif provider == 'openrouter':
            if not data.get('OPENROUTER_API_KEY'): logger.warning("LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is missing.")
            if not data.get('OPENROUTER_MODEL_NAME'): logger.warning("LLM_PROVIDER is 'openrouter' but OPENROUTER_MODEL_NAME is missing (default will be used).")
        elif provider == 'local':
            if not data.get('LOCAL_LLM_MODEL_NAME'): logger.warning("LLM_PROVIDER is 'local' but LOCAL_LLM_MODEL_NAME is missing (default will be used).")
            if not data.get('LOCAL_LLM_API_BASE_URL'): logger.warning("LLM_PROVIDER is 'local' but LOCAL_LLM_API_BASE_URL is missing (default will be used).")

        user_agent = data.get('GEOCODING_USER_AGENT', '')
        if not user_agent or 'your_email@example.com' in user_agent or 'example@example.com' in user_agent or not '@' in user_agent:
             logger.warning("GEOCODING_USER_AGENT is not set, uses a default example, or lacks contact info. Please update it in your .env file as per Nominatim's policy.")
        return data

try:
    settings = Settings()
    # Log crucial settings (avoiding full keys)
    logger.info(f"Settings loaded. Log Level: {settings.log_level}, LLM Provider: {settings.llm_provider}")
    if settings.llm_provider == 'google': logger.info(f"Google Model: {settings.google_model_name}")
    elif settings.llm_provider == 'openrouter': logger.info(f"OpenRouter Model: {settings.openrouter_model_name}")
    elif settings.llm_provider == 'local': logger.info(f"Local LLM Model: {settings.local_llm_model_name}, URL: {settings.local_llm_api_base_url}")
    logger.info(f"Geocoding User Agent: {settings.geocoding_user_agent}")

except Exception as e:
     logger.critical(f"CRITICAL ERROR: Failed to load settings: {e}", exc_info=True)
     # Fallback to basic settings if loading fails, to allow app to at least start and show error
     class FallbackSettings:
         log_level="ERROR"; llm_provider="none"; geocoding_user_agent="FallbackAgent/0.0"; streamlit_server_port=8501
         google_model_options = ["gemini-1.5-flash-latest"] # Provide a minimal list
     settings = FallbackSettings()
     print(f"CRITICAL: Using fallback settings due to error: {e}")
     # raise SystemExit(f"Failed to initialize settings: {e}") from e # Or allow to run with fallback