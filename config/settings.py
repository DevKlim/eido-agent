import logging
import os
from typing import List, Optional, Literal, Any, ClassVar
from pydantic import field_validator, model_validator, Field, BaseModel, ConfigDict as PydanticConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

settings_logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

# --- Fallback settings class using BaseModel to PREVENT environment loading ---
# This ensures that if the main settings fail, this fallback is pure and safe.


class FallbackSettings(BaseModel):
    app_name: str = "EIDO Sentinel (Fallback)"
    api_base_url: str = "http://localhost:8000"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_server_port: int = 8501
    database_url: Optional[str] = None
    log_level: Literal['DEBUG', 'INFO',
                       'WARNING', 'ERROR', 'CRITICAL'] = "INFO"
    llm_provider: Literal['google', 'openrouter', 'local', 'none'] = "none"
    google_api_key: Optional[str] = None
    google_model_name: str = "gemini-2.5-flash-preview-05-02"
    openrouter_api_key: Optional[str] = None
    openrouter_model_name: Optional[str] = "openai/gpt-4o-mini"
    openrouter_api_base_url: str = "https://openrouter.ai/api/v1"
    local_llm_api_base_url: Optional[str] = "http://localhost:11434/v1"
    local_llm_model_name: Optional[str] = "llama3:latest"
    local_llm_api_key: Optional[str] = "ollama"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    geocoding_user_agent: str = "EidoSentinelApp/0.9 (klimentlamh@gmail.com)"
    similarity_threshold: float = 0.70
    time_window_minutes: int = 60
    distance_threshold_km: float = 1.0
    streamlit_auto_shutdown_delay: int = 0
    model_config = PydanticConfigDict(extra='ignore')


class Settings(BaseSettings):
    """
    Application settings.
    Values are loaded from environment variables and/or a .env file.
    """
    @field_validator('*', mode='before')
    @classmethod
    def strip_quotes_and_whitespace(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().strip('"').strip("'").strip()
        return v

    # --- Application Core ---
    app_name: str = Field("EIDO Sentinel", validation_alias='APP_NAME')
    api_base_url: str = Field("http://localhost:8000",
                              validation_alias='API_BASE_URL')
    # CRITICAL CHANGE: Default host is now 0.0.0.0 for container-readiness.
    api_host: str = Field("0.0.0.0", validation_alias='API_HOST')
    api_port: int = Field(8000, validation_alias='API_PORT')
    streamlit_server_port: int = Field(
        8501, validation_alias='STREAMLIT_SERVER_PORT')
    database_url: Optional[str] = Field(None, validation_alias='DATABASE_URL')
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR',
                       'CRITICAL'] = Field("INFO", validation_alias='LOG_LEVEL')
    llm_provider: Literal['google', 'openrouter', 'local', 'none'] = Field(
        "google", validation_alias='LLM_PROVIDER')
    google_api_key: Optional[str] = Field(
        None, validation_alias='GOOGLE_API_KEY')
    google_model_name: str = Field(
        "gemini-2.5-flash-05-20", validation_alias='GOOGLE_MODEL_NAME')
    openrouter_api_key: Optional[str] = Field(
        None, validation_alias='OPENROUTER_API_KEY')
    openrouter_model_name: Optional[str] = Field(
        "openai/gpt-4o-mini", validation_alias='OPENROUTER_MODEL_NAME')
    openrouter_api_base_url: str = Field(
        "https://openrouter.ai/api/v1", validation_alias='OPENROUTER_API_BASE_URL')
    local_llm_api_base_url: Optional[str] = Field(
        "http://localhost:11434/v1", validation_alias='LOCAL_LLM_API_BASE_URL')
    local_llm_model_name: Optional[str] = Field(
        "llama3:latest", validation_alias='LOCAL_LLM_MODEL_NAME')
    local_llm_api_key: Optional[str] = Field(
        "ollama", validation_alias='LOCAL_LLM_API_KEY')
    embedding_model_name: str = Field(
        "all-MiniLM-L6-v2", validation_alias='EMBEDDING_MODEL_NAME')
    geocoding_user_agent: str = Field(
        "EidoSentinelApp/0.9.1 (contact: your_email@example.com)", validation_alias='GEOCODING_USER_AGENT')
    similarity_threshold: float = Field(
        0.70, ge=0.0, le=1.0, validation_alias='SIMILARITY_THRESHOLD')
    time_window_minutes: int = Field(
        60, gt=0, validation_alias='TIME_WINDOW_MINUTES')
    distance_threshold_km: float = Field(
        1.0, gt=0, validation_alias='DISTANCE_THRESHOLD_KM')
    streamlit_auto_shutdown_delay: int = Field(
        0, validation_alias='STREAMLIT_AUTO_SHUTDOWN_DELAY')

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False
    )

    @model_validator(mode='after')
    def check_dependencies(self) -> 'Settings':
        # --- LLM Provider Checks ---
        if self.llm_provider == 'google' and not self.google_api_key:
            settings_logger.warning(
                "LLM_PROVIDER is 'google' but GOOGLE_API_KEY is not set.")
        if self.llm_provider == 'openrouter' and not self.openrouter_api_key:
            settings_logger.warning(
                "LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set.")

        # --- Database URL Checks ---
        if not self.database_url:
            settings_logger.error(
                "CRITICAL: DATABASE_URL is not set. The application cannot connect to the database.")
        else:
            # FIX: Automatically correct the postgresql scheme for async compatibility
            if self.database_url.startswith("postgresql://"):
                settings_logger.warning(
                    "DATABASE_URL uses 'postgresql://' scheme. For async support, it will be automatically changed to 'postgresql+asyncpg://'."
                )
                self.database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
                settings_logger.info(f"Corrected DATABASE_URL to: {self.database_url}")
            elif self.database_url.startswith("postgresql") and not self.database_url.startswith("postgresql+asyncpg://"):
                settings_logger.critical(
                    "DATABASE_URL uses an unrecognized PostgreSQL scheme. It must start with 'postgresql+asyncpg://'. "
                    f"Current value: {self.database_url}"
                )
            elif self.database_url.startswith("sqlite"):
                settings_logger.info(
                    "Using SQLite database for local development.")
            else:
                settings_logger.warning(
                    f"Unrecognized DATABASE_URL scheme: {self.database_url}")

        return self


# Singleton instance
try:
    settings = Settings()
    settings_logger.info(
        f"Application settings loaded successfully. LLM Provider: {settings.llm_provider.upper()}")
except Exception as e:
    settings_logger.critical(
        f"CRITICAL ERROR: Failed to load application settings from environment: {e}", exc_info=False)
    settings_logger.warning(
        "Falling back to default, safe settings. Application may not function correctly.")
    settings = FallbackSettings()