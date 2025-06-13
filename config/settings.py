import logging
import os
from typing import List, Optional, Literal, Union, Any, ClassVar # Added ClassVar
from pydantic import field_validator, model_validator, Field, ValidationInfo # Added ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging for the settings module itself
settings_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config for early log messages


class Settings(BaseSettings):
    """
    Application settings.
    Values are loaded from environment variables and/or a .env file.
    """
    # --- Application Core ---
    app_name: str = Field("EIDO Sentinel", validation_alias='APP_NAME')
    api_base_url: str = Field("http://localhost:8000", validation_alias='API_BASE_URL')
    api_host: str = Field("127.0.0.1", validation_alias='API_HOST')
    api_port: int = Field(8000, validation_alias='API_PORT')
    streamlit_server_port: int = Field(8501, validation_alias='STREAMLIT_SERVER_PORT')

    # --- Database ---
    database_url: str = Field(
        "postgresql+asyncpg://user:password@localhost:5432/eido_sentinel_db",
        validation_alias='DATABASE_URL'
    )

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
    google_model_name: str = Field("gemini-2.5-flash-preview-05-20", validation_alias='GOOGLE_MODEL_NAME')
    
    # Supported Google models (examples, check Google Cloud for latest)
    _google_model_options: ClassVar[List[str]] = [ # Changed to ClassVar[List[str]]
        "gemini-2.5-flash-preview-05-20", "gemini-2.0-flash" # Original user's value kept, validator handles warning
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
    geocoding_user_agent: str = Field("EidoSentinelApp/0.9 (contact: your_email@example.com)", validation_alias='GEOCODING_USER_AGENT')

    # --- Incident Matching ---
    similarity_threshold: float = Field(0.70, ge=0.0, le=1.0, validation_alias='SIMILARITY_THRESHOLD')
    time_window_minutes: int = Field(60, gt=0, validation_alias='TIME_WINDOW_MINUTES')
    distance_threshold_km: float = Field(1.0, gt=0, validation_alias='DISTANCE_THRESHOLD_KM')

    # --- Streamlit UI Auto Shutdown ---
    streamlit_auto_shutdown_delay: int = Field(0, validation_alias='STREAMLIT_AUTO_SHUTDOWN_DELAY')


    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local", ".env.prod"), # Load from .env files
        env_file_encoding='utf-8',
        extra='ignore', # Ignore extra fields from .env
        case_sensitive=False # Environment variables are typically case-insensitive
    )

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if value.upper() not in allowed_levels:
            raise ValueError(f"Invalid log_level: {value}. Must be one of {allowed_levels}")
        return value.upper()

    @field_validator('google_model_name')
    @classmethod
    def validate_google_model_name(cls, value: str, info: ValidationInfo) -> str:
        # Accessing _google_model_options via cls should now work correctly due to ClassVar.
        if value not in cls._google_model_options: # type: ignore
            settings_logger.warning(
                f"GOOGLE_MODEL_NAME '{value}' is not in the predefined list of common models: "
                f"{cls._google_model_options}. Ensure it's a valid and available model for your API key." # type: ignore
            )
        return value

    @model_validator(mode='before')
    @classmethod
    def check_llm_dependencies(cls, values: Any) -> Any: # Changed type hint for values
        if not isinstance(values, dict): # Check if values is a dict, common in 'before' mode
            return values

        def get_value_from_input_or_env(key_name: str, field_alias: Optional[str], default: Optional[Any] = None) -> Optional[Any]:
            # In 'before' mode, 'values' is the raw input data (e.g., from .env or direct instantiation)
            # Aliases are used as keys if present in .env or initial data
            # Pydantic's BaseSettings has complex loading: .env -> os.environ -> defaults
            # This validator runs *before* Pydantic fully resolves aliases and applies defaults.
            
            # 1. Check if the alias is in the raw input 'values' (e.g., from .env)
            if field_alias and field_alias in values:
                return values[field_alias]
            
            # 2. Check if the field name itself is in raw input 'values' (e.g., direct instantiation)
            if key_name in values:
                return values[key_name]

            # 3. Check actual environment variables (Pydantic might not have loaded them into 'values' yet)
            # Use alias if available, otherwise uppercase field name.
            env_var_to_check = field_alias if field_alias else key_name.upper()
            env_val = os.environ.get(env_var_to_check)
            if env_val is not None:
                return env_val
            
            # 4. Fallback to default (or None if no default provided)
            # This is tricky because Pydantic's own default application happens later.
            # For simple presence checks (like API keys), it's often enough to see if it's None.
            return default

        llm_provider_alias = cls.model_fields['llm_provider'].validation_alias
        llm_provider_val = get_value_from_input_or_env('llm_provider', str(llm_provider_alias) if llm_provider_alias else None, 'google')
        llm_provider = str(llm_provider_val).lower() if llm_provider_val is not None else 'google'

        if llm_provider == 'google':
            google_api_key_alias = cls.model_fields['google_api_key'].validation_alias
            if not get_value_from_input_or_env('google_api_key', str(google_api_key_alias) if google_api_key_alias else None):
                settings_logger.warning("LLM_PROVIDER is 'google' but GOOGLE_API_KEY is not set.")
        elif llm_provider == 'openrouter':
            openrouter_api_key_alias = cls.model_fields['openrouter_api_key'].validation_alias
            if not get_value_from_input_or_env('openrouter_api_key', str(openrouter_api_key_alias) if openrouter_api_key_alias else None):
                settings_logger.warning("LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set.")
        elif llm_provider == 'local':
            local_llm_api_base_url_alias = cls.model_fields['local_llm_api_base_url'].validation_alias
            if not get_value_from_input_or_env('local_llm_api_base_url', str(local_llm_api_base_url_alias) if local_llm_api_base_url_alias else None):
                settings_logger.warning("LLM_PROVIDER is 'local' but LOCAL_LLM_API_BASE_URL is not set or is empty.")
        elif llm_provider == 'none':
            settings_logger.info("LLM_PROVIDER is 'none'. LLM-dependent features will be disabled.")
        else:
            settings_logger.error(f"Invalid LLM_PROVIDER: {llm_provider_val}. Check .env file.")

        geocoding_user_agent_alias = cls.model_fields['geocoding_user_agent'].validation_alias
        geocoding_agent = get_value_from_input_or_env('geocoding_user_agent', str(geocoding_user_agent_alias) if geocoding_user_agent_alias else None, '')
        geocoding_agent_str = str(geocoding_agent) if geocoding_agent is not None else ''
        if not geocoding_agent_str or "@" not in geocoding_agent_str or "/" not in geocoding_agent_str:
            settings_logger.critical(
                "GEOCODING_USER_AGENT is missing, invalid, or does not contain an email and app name/version. "
                "This is REQUIRED by Nominatim's usage policy. Service may be blocked. "
                "Example: 'MyCoolApp/1.0 (contact@example.com)'"
            )

        database_url_alias = cls.model_fields['database_url'].validation_alias
        db_url = get_value_from_input_or_env('database_url', str(database_url_alias) if database_url_alias else None, '')
        db_url_str = str(db_url) if db_url is not None else ''
        if not db_url_str.startswith("postgresql+asyncpg://"):
            settings_logger.critical(
                "DATABASE_URL does not start with 'postgresql+asyncpg://'. "
                "This is required for async database operations with SQLAlchemy and FastAPI. "
                f"Current value: {db_url_str}"
            )
        return values

# Singleton instance
try:
    settings = Settings()
    settings_logger.info(f"Application settings loaded. LLM Provider: {settings.llm_provider.upper()}, Log Level: {settings.log_level}")
except Exception as e:
    settings_logger.critical(f"CRITICAL ERROR: Failed to load application settings: {e}", exc_info=True)
    settings_logger.warning("Falling back to default settings. Application may not function correctly.")
    # Create a default settings object so the application can at least try to import it
    class FallbackSettings(BaseSettings): # Define a minimal fallback
        app_name: str = "EIDO Sentinel (Fallback)"
        api_base_url: str = "http://localhost:8000"
        api_host: str = "127.0.0.1"
        api_port: int = 8000
        streamlit_server_port: int = 8501
        database_url: str = "postgresql+asyncpg://user:password@localhost:5432/eido_sentinel_db" 
        log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = "INFO"
        llm_provider: Literal['google', 'openrouter', 'local', 'none'] = "none" 
        google_api_key: Optional[str] = None
        google_model_name: str = "gemini-1.0-pro" # A known valid default
        openrouter_api_key: Optional[str] = None
        openrouter_model_name: Optional[str] = "openai/gpt-4o-mini"
        openrouter_api_base_url: str = "https://openrouter.ai/api/v1"
        local_llm_api_base_url: Optional[str] = "http://localhost:11434/v1"
        local_llm_model_name: Optional[str] = "llama3:latest"
        local_llm_api_key: Optional[str] = "ollama"
        embedding_model_name: str = "all-MiniLM-L6-v2"
        geocoding_user_agent: str = "EidoSentinelApp/0.9 (fallback_contact@example.com)"
        similarity_threshold: float = 0.70
        time_window_minutes: int = 60
        distance_threshold_km: float = 1.0
        streamlit_auto_shutdown_delay: int = 0

        model_config = SettingsConfigDict(extra='ignore')
    settings = FallbackSettings()