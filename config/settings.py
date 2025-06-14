import logging
import os
from typing import List, Optional, Literal, Any, ClassVar
from pydantic import field_validator, model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

settings_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Settings(BaseSettings):
    """
    Application settings.
    Values are loaded from environment variables and/or a .env file.
    """
    # --- Helper function to strip quotes ---
    @field_validator('*', mode='before')
    @classmethod
    def strip_quotes_and_whitespace(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().strip('"').strip("'")
        return v

    # --- Application Core ---
    app_name: str = Field("EIDO Sentinel", validation_alias='APP_NAME')
    api_base_url: str = Field("http://localhost:8000", validation_alias='API_BASE_URL')
    api_host: str = Field("127.0.0.1", validation_alias='API_HOST')
    api_port: int = Field(8000, validation_alias='API_PORT')
    streamlit_server_port: int = Field(8501, validation_alias='STREAMLIT_SERVER_PORT')

    # --- Database ---
    database_url: Optional[str] = Field(None, validation_alias='DATABASE_URL')

    # --- Logging ---
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        "INFO", validation_alias='LOG_LEVEL'
    )

    # --- LLM Configuration ---
    llm_provider: Literal['google', 'openrouter', 'local', 'none'] = Field(
        "google", validation_alias='LLM_PROVIDER'
    )

    # Google Generative AI
    google_api_key: Optional[str] = Field(None, validation_alias='GOOGLE_API_KEY')
    google_model_name: str = Field("gemini-1.5-flash", validation_alias='GOOGLE_MODEL_NAME')
    
    _google_model_options: ClassVar[List[str]] = [
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"
    ]

    # OpenRouter
    openrouter_api_key: Optional[str] = Field(None, validation_alias='OPENROUTER_API_KEY')
    openrouter_model_name: Optional[str] = Field("openai/gpt-4o-mini", validation_alias='OPENROUTER_MODEL_NAME')
    openrouter_api_base_url: str = Field("https://openrouter.ai/api/v1", validation_alias='OPENROUTER_API_BASE_URL')

    # Local LLM (Ollama, LM Studio etc.)
    local_llm_api_base_url: Optional[str] = Field("http://localhost:11434/v1", validation_alias='LOCAL_LLM_API_BASE_URL')
    local_llm_model_name: Optional[str] = Field("llama3:latest", validation_alias='LOCAL_LLM_MODEL_NAME')
    local_llm_api_key: Optional[str] = Field("ollama", validation_alias='LOCAL_LLM_API_KEY')

    # --- Embedding Service ---
    embedding_model_name: str = Field("all-MiniLM-L6-v2", validation_alias='EMBEDDING_MODEL_NAME')

    # --- Geocoding Service (Nominatim) ---
    geocoding_user_agent: str = Field("EidoSentinelApp/0.9.1 (contact: your_email@example.com)", validation_alias='GEOCODING_USER_AGENT')

    # --- Incident Matching ---
    similarity_threshold: float = Field(0.70, ge=0.0, le=1.0, validation_alias='SIMILARITY_THRESHOLD')
    time_window_minutes: int = Field(60, gt=0, validation_alias='TIME_WINDOW_MINUTES')
    distance_threshold_km: float = Field(1.0, gt=0, validation_alias='DISTANCE_THRESHOLD_KM')

    # --- Streamlit UI Auto Shutdown ---
    streamlit_auto_shutdown_delay: int = Field(0, validation_alias='STREAMLIT_AUTO_SHUTDOWN_DELAY')

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False
    )
    
    @model_validator(mode='after')
    def check_dependencies(self) -> 'Settings':
        if self.llm_provider == 'google' and not self.google_api_key:
            settings_logger.warning("LLM_PROVIDER is 'google' but GOOGLE_API_KEY is not set.")
        
        if self.llm_provider == 'openrouter' and not self.openrouter_api_key:
            settings_logger.warning("LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set.")
            
        if self.llm_provider == 'local' and not self.local_llm_api_base_url:
            settings_logger.warning("LLM_PROVIDER is 'local' but LOCAL_LLM_API_BASE_URL is not set.")
        
        if "example.com" in self.geocoding_user_agent or "@" not in self.geocoding_user_agent:
            settings_logger.critical(
                "GEOCODING_USER_AGENT is not set correctly. This is required by Nominatim's policy."
            )
            
        if self.database_url and not self.database_url.startswith("postgresql+asyncpg://"):
            settings_logger.critical(
                f"DATABASE_URL does not start with 'postgresql+asyncpg://'. Current value: {self.database_url}"
            )
        elif not self.database_url:
             settings_logger.error("DATABASE_URL is not set. The application will not be able to connect to the database.")

        return self

# Singleton instance
try:
    settings = Settings()
    settings_logger.info(f"Application settings loaded. LLM Provider: {settings.llm_provider.upper()}, Log Level: {settings.log_level}")
except Exception as e:
    settings_logger.critical(f"CRITICAL ERROR: Failed to load application settings: {e}", exc_info=True)
    settings_logger.warning("Falling back to default settings. Application may not function correctly.")
    # Fallback only if all else fails, so the app can at least import the module
    settings = Settings(_env_file=None)