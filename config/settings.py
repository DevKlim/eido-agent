# config/settings.py
import os
import logging
from pydantic import Field, field_validator, model_validator, ValidationError, FieldValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal

# --- Environment File Loading ---
# Determine the path to the .env file relative to this settings.py file
# settings.py is in config/, .env is in the parent directory (project root)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

# Check CWD and if .env exists for debugging
print(f"--- CWD in settings.py: {os.getcwd()} ---")
print(f"--- .env exists here?: {os.path.exists(dotenv_path)} ---")

# Load .env file using the determined path
from dotenv import load_dotenv
load_dotenv(dotenv_path=dotenv_path)

# --- Logger Setup (Basic) ---
# Configure logging early so subsequent messages during settings load are captured
# A more sophisticated setup might involve configuring handlers later
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


# --- Application Settings Model ---

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and/or .env file.
    Uses pydantic-settings for automatic loading and validation.
    """
    # --- General Settings ---
    app_name: str = Field("EIDO Sentinel POC", description="Name of the application.")
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        description="Logging level for the application."
    )

    # --- LLM Configuration ---
    llm_provider: Optional[Literal['google', 'openrouter', 'none']] = Field(
        default='none', # Default to 'none' if not set
        description="The LLM provider to use ('google', 'openrouter', or 'none')."
    )
    # Google Specific
    google_api_key: Optional[str] = Field(None, alias='GOOGLE_API_KEY', description="API Key for Google Generative AI.")
    google_model_name: str = Field("gemini-1.5-flash-latest", description="Specific Google model name to use.")
    # OpenRouter / Generic OpenAI-compatible Specific
    llm_api_key: Optional[str] = Field(None, alias='LLM_API_KEY', description="API Key for OpenRouter or other generic LLM provider.")
    llm_api_base_url: Optional[str] = Field(None, alias='LLM_API_BASE_URL', description="Base URL for OpenRouter or other generic LLM provider.")
    llm_model_name: Optional[str] = Field(None, alias='LLM_MODEL_NAME', description="Model name for OpenRouter or other generic LLM provider (e.g., 'openai/gpt-4o-mini').")

    # --- Embedding Model ---
    embedding_model_name: str = Field("all-MiniLM-L6-v2", description="Sentence Transformer model for embeddings.")

    # --- Geocoding Service ---
    geocoding_user_agent: str = Field("EidoAgentPOC/1.0 (your_email@example.com)", description="User agent string for Nominatim geocoding requests. **Please change the email.**")

    # --- Incident Matching Parameters ---
    time_window_minutes: int = Field(60, description="Time window (minutes) for considering reports part of the same incident.")
    distance_threshold_km: float = Field(1.0, description="Distance threshold (km) for matching incidents by location.")
    similarity_threshold: float = Field(0.70, ge=0.0, le=1.0, description="Minimum score (0-1) for matching incidents based on combined factors.")


    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        # `.env` file encoding.
        env_file_encoding='utf-8',
        # Specifies the path to the .env file relative to the project root
        # This might be redundant if load_dotenv is called explicitly above, but good practice.
        env_file=dotenv_path,
        # Allow extra fields not defined in the model? Usually False for strictness.
        extra='ignore'
    )

    # --- Custom Validators ---
    @field_validator('log_level', mode='before')
    @classmethod
    def uppercase_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator('geocoding_user_agent')
    @classmethod
    def check_user_agent_email(cls, value: str) -> str:
        if 'your_email@example.com' in value:
            logger.warning("Using default geocoding user agent email. Please update 'geocoding_user_agent' in your .env file or environment variables with your actual email for Nominatim TOS compliance.")
        return value

    # Model validator to check conditional requirements (e.g., API keys based on provider)
    @model_validator(mode='after')
    def check_llm_config(self) -> 'Settings':
        provider = self.llm_provider
        logger.info(f"Validating LLM configuration for provider: '{provider}'")

        if provider == 'google':
            if not self.google_api_key:
                logger.warning("LLM provider is 'google' but GOOGLE_API_KEY is not set. LLM calls will fail.")
                # raise ValueError("GOOGLE_API_KEY is required when llm_provider is 'google'")
            if not self.google_model_name:
                 logger.warning("LLM provider is 'google' but google_model_name is not set. Using default.")
                 # Default is already set via Field, but could raise error if needed:
                 # raise ValueError("google_model_name is required when llm_provider is 'google'")

        elif provider == 'openrouter':
            if not self.llm_api_key:
                 logger.warning("LLM provider is 'openrouter' but LLM_API_KEY is not set. LLM calls will fail.")
                 # raise ValueError("LLM_API_KEY is required when llm_provider is 'openrouter'")
            if not self.llm_api_base_url:
                 logger.warning("LLM provider is 'openrouter' but LLM_API_BASE_URL is not set. LLM calls will fail.")
                 # raise ValueError("LLM_API_BASE_URL is required when llm_provider is 'openrouter'")
            if not self.llm_model_name:
                 logger.warning("LLM provider is 'openrouter' but LLM_MODEL_NAME is not set. LLM calls will fail.")
                 # raise ValueError("LLM_MODEL_NAME is required when llm_provider is 'openrouter'")

        elif provider == 'none':
             logger.info("LLM provider is set to 'none'. LLM summarization/recommendation features will be disabled.")

        else:
             # This case should be prevented by Literal validation, but good as a safeguard
             logger.error(f"Unsupported llm_provider configured: '{provider}'. LLM features will be disabled.")
             # Force provider to 'none' if invalid to prevent errors downstream
             # self.llm_provider = 'none' # This mutation isn't allowed in model_validator

        return self


# --- Load Settings Instance ---
try:
    settings = Settings()
    # Optionally print loaded settings for verification (careful with secrets)
    # logger.debug(f"Settings loaded: {settings.model_dump(exclude={'google_api_key', 'llm_api_key'})}")
    logger.info("Application settings loaded successfully.")
except ValidationError as e:
    logger.critical(f"CRITICAL ERROR: Failed to load or validate settings: {e}", exc_info=True)
    # Decide how to handle this - exit, use defaults, etc.
    # Forcing exit might be safest if settings are crucial.
    raise SystemExit(f"Settings validation failed: {e}") from e
except Exception as e:
     logger.critical(f"CRITICAL ERROR: An unexpected error occurred while loading settings: {e}", exc_info=True)
     raise SystemExit(f"Unexpected settings error: {e}") from e