import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator, PostgresDsn
from typing import List, Optional, Literal, Dict, Any
import logging

logger = logging.getLogger(__name__)

LlmProvider = Literal['google', 'openrouter', 'local', 'none']

# Updated Google Model Options
GOOGLE_MODEL_OPTIONS = [
    "gemini-2.0-flash", # Set as default
    "gemini-1.5-flash-latest", # Previous default, still an option
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash-lite",
    "gemini-1.0-pro", # Stable model
    # Add other models as needed
]

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

class Settings(BaseSettings):
    app_name: str = "EIDO Sentinel"
    log_level: str = Field("INFO", validation_alias='LOG_LEVEL')

    # --- New Settings ---
    database_url: PostgresDsn = Field(..., validation_alias='DATABASE_URL')
    api_base_url: str = Field("http://localhost:8000", validation_alias='API_BASE_URL')
    # For Render deployment, API_BASE_URL should be the *.onrender.com URL

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
    
    geocoding_user_agent: str = Field("EidoSentinelApp/0.9 (contact: your_email@example.com)", validation_alias='GEOCODING_USER_AGENT')

    similarity_threshold: float = Field(0.70, validation_alias='SIMILARITY_THRESHOLD')
    time_window_minutes: int = Field(60, validation_alias='TIME_WINDOW_MINUTES')
    distance_threshold_km: float = Field(1.0, validation_alias='DISTANCE_THRESHOLD_KM')

    api_host: str = Field("127.0.0.1", validation_alias='API_HOST')
    api_port: int = Field(8000, validation_alias='API_PORT')

    streamlit_server_port: int = Field(8503, validation_alias='STREAMLIT_SERVER_PORT') # Changed default port

    model_config = SettingsConfigDict(
        env_file=('.env', '.env.production', '.env.development'), 
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
    def validate_google_model_name(cls, v: str, info: Any): 
        if v not in GOOGLE_MODEL_OPTIONS:
             logger.warning(f"Configured google_model_name '{v}' is not in the predefined GOOGLE_MODEL_OPTIONS. This might be intentional for a custom/newer model. Using '{v}'.")
        return v
        
    @model_validator(mode='before') 
    @classmethod
    def check_llm_dependencies(cls, data: Any) -> Any:
        if not isinstance(data, dict): return data 

        provider = data.get('LLM_PROVIDER', data.get('llm_provider', 'none'))
        
        if provider == 'google' and not data.get('GOOGLE_API_KEY', data.get('google_api_key')):
            logger.warning("LLM_PROVIDER is 'google' but GOOGLE_API_KEY is missing.")
        elif provider == 'openrouter':
            if not data.get('OPENROUTER_API_KEY', data.get('openrouter_api_key')): 
                logger.warning("LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is missing.")
            if not data.get('OPENROUTER_MODEL_NAME', data.get('openrouter_model_name')): 
                logger.warning("LLM_PROVIDER is 'openrouter' but OPENROUTER_MODEL_NAME is missing (default will be used).")
        elif provider == 'local':
            if not data.get('LOCAL_LLM_MODEL_NAME', data.get('local_llm_model_name')): 
                logger.warning("LLM_PROVIDER is 'local' but LOCAL_LLM_MODEL_NAME is missing (default will be used).")
            if not data.get('LOCAL_LLM_API_BASE_URL', data.get('local_llm_api_base_url')): 
                logger.warning("LLM_PROVIDER is 'local' but LOCAL_LLM_API_BASE_URL is missing (default will be used).")

        user_agent = data.get('GEOCODING_USER_AGENT', data.get('geocoding_user_agent', ''))
        if not user_agent or 'your_email@example.com' in user_agent or 'example@example.com' in user_agent or not '@' in user_agent:
             logger.warning("GEOCODING_USER_AGENT is not set, uses a default example, or lacks contact info. Please update it in your .env file as per Nominatim's policy.")
        
        # Validate DATABASE_URL format for asyncpg
        db_url = data.get('DATABASE_URL', data.get('database_url'))
        if db_url and not db_url.startswith("postgresql+asyncpg://"):
            logger.warning(f"DATABASE_URL '{db_url}' does not start with 'postgresql+asyncpg://'. Ensure it's correct for async operations.")
        elif not db_url:
            logger.error("DATABASE_URL is not set. Database operations will fail.")

        return data

try:
    settings = Settings()
    logger.info(f"Settings loaded. Log Level: {settings.log_level}, LLM Provider: {settings.llm_provider}")
    logger.info(f"Database URL: {str(settings.database_url).split('@')[-1] if settings.database_url else 'Not Set'}") # Mask credentials
    logger.info(f"API Base URL: {settings.api_base_url}")
    if settings.llm_provider == 'google': logger.info(f"Google Model: {settings.google_model_name}")
    elif settings.llm_provider == 'openrouter': logger.info(f"OpenRouter Model: {settings.openrouter_model_name}")
    elif settings.llm_provider == 'local': logger.info(f"Local LLM Model: {settings.local_llm_model_name}, URL: {settings.local_llm_api_base_url}")
    logger.info(f"Geocoding User Agent: {settings.geocoding_user_agent}")
    logger.info(f"Default Streamlit Port (for reference): {settings.streamlit_server_port}")


except Exception as e:
     logger.critical(f"CRITICAL ERROR: Failed to load settings: {e}", exc_info=True)
     class FallbackSettings: # type: ignore
         log_level="ERROR"; llm_provider="none"; geocoding_user_agent="FallbackAgent/0.0"; streamlit_server_port=8503 # Changed
         api_host="127.0.0.1"; api_port=8000;
         google_model_options = ["gemini-2.0-flash"] 
         google_model_name = "gemini-2.0-flash"
         database_url = "postgresql+asyncpg://user:pass@host/db" # Dummy
         api_base_url = "http://localhost:8000" # Dummy
     settings = FallbackSettings() # type: ignore
     print(f"CRITICAL: Using fallback settings due to error: {e}")