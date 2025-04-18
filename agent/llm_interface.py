# agent/llm_interface.py
import logging
from typing import List, Optional
import sys
import json

# Import settings INSTANCE and necessary LLM client libraries
try:
    # --->>> CHANGE IMPORT HERE <<<---
    # Import the 'settings' instance directly, AFTER it's created in config/settings.py
    from config.settings import settings
    # --->>> END CHANGE <<<---

    # Log provider AFTER import succeeds using the instance
    llm_provider_setting = getattr(settings, 'llm_provider', 'Attribute Missing')
    print(f"--- Settings Instance Accessed in llm_interface.py ---")
    print(f"--- Settings Instance: LLM Provider = {llm_provider_setting} ---")

except ImportError as e:
     print(f"--- FAILED to import settings INSTANCE in llm_interface.py: {e} ---")
     # This likely means config.settings itself failed earlier
     raise SystemExit(f"llm_interface failed to import settings instance: {e}") from e
except AttributeError as e:
     # This might happen if 'settings' isn't defined in config.settings (shouldn't occur)
     print(f"--- FAILED: 'settings' instance not found in config.settings module?: {e} ---")
     raise SystemExit(f"llm_interface failed to find settings instance: {e}") from e


from data_models.schemas import ReportCoreData

# Conditional imports based on provider
llm_client = None
model_name = None
# Get provider safely from the imported settings instance
llm_provider = getattr(settings, 'llm_provider', 'none')

logger = logging.getLogger(__name__)

# --- LLM Client Initialization (remains the same logic, but uses the loaded settings) ---
try:
    logger.info(f"Attempting to configure LLM client for provider: '{llm_provider}'")

    if llm_provider == 'google':
        import google.generativeai as genai
        # Access attributes directly from the settings instance
        google_api_key = settings.google_api_key
        google_model_name = settings.google_model_name
        if google_api_key:
            genai.configure(api_key=google_api_key)
            llm_client = genai.GenerativeModel(google_model_name)
            model_name = google_model_name
            logger.info(f"Configured Google Generative AI client for model: {model_name}")
        else:
            logger.error("Google provider selected but GOOGLE_API_KEY is missing or not loaded.")

    elif llm_provider == 'openrouter':
        from openai import OpenAI
        openrouter_api_key = settings.llm_api_key
        openrouter_base_url = settings.llm_api_base_url
        openrouter_model_name = settings.llm_model_name

        if openrouter_api_key and openrouter_base_url and openrouter_model_name:
            llm_client = OpenAI(
                api_key=openrouter_api_key,
                base_url=openrouter_base_url
            )
            model_name = openrouter_model_name
            logger.info(f"Configured OpenAI client for OpenRouter model: {model_name} at {openrouter_base_url}")
        else:
             missing_configs = []
             if not openrouter_api_key: missing_configs.append("LLM_API_KEY")
             if not openrouter_base_url: missing_configs.append("LLM_API_BASE_URL")
             if not openrouter_model_name: missing_configs.append("LLM_MODEL_NAME")
             logger.error(f"OpenRouter provider selected but required config(s) are missing: {', '.join(missing_configs)}")

    elif llm_provider == 'none':
         logger.info("LLM provider set to 'none'. LLM features disabled.")
         llm_client = None

    else:
        logger.error(f"Unsupported LLM provider configured: {llm_provider}")
        llm_client = None

except ImportError as e:
     logger.critical(f"Failed to import LLM library for provider '{llm_provider}': {e}. Please install required packages (e.g., 'pip install google-generativeai' or 'pip install openai').", exc_info=True)
     llm_client = None
except AttributeError as e:
     # Catch errors if settings attributes (like google_api_key) are missing AFTER import
     logger.critical(f"Configuration Error: Missing expected attribute in the loaded settings object for provider '{llm_provider}'. Error: {e}", exc_info=True)
     llm_client = None
except Exception as e:
     logger.critical(f"Failed to configure LLM client for provider '{llm_provider}'. Error details: {e}", exc_info=True)
     llm_client = None

# Log final client status
if llm_client:
    logger.info(f"LLM Client for '{llm_provider}' appears to be configured.")
else:
    logger.warning(f"LLM Client for '{llm_provider}' is NOT configured or failed to initialize. LLM calls will be skipped.")


# --- _call_llm, summarize_incident, recommend_actions functions remain unchanged ---
# (Keep the enhanced logging added previously in _call_llm)

def _call_llm(prompt: str) -> Optional[str]:
    """Internal function to call the configured LLM."""
    # --- >>> ENHANCED LOGGING <<< ---
    logger.debug(f"--- Attempting LLM call (_call_llm) ---")
    if not llm_client:
        logger.error("LLM client is not configured or failed to initialize. Cannot call LLM.")
        return None
    if not llm_provider or llm_provider == 'none':
         logger.warning(f"LLM call skipped because provider is '{llm_provider}'.")
         return None
    if not model_name:
         logger.error(f"LLM call skipped because model name is not set for provider '{llm_provider}'.")
         return None

    logger.info(f"Calling LLM Provider: {llm_provider}, Model: {model_name}")
    # Log prompt safely (consider truncation for very long prompts in production)
    prompt_log_max_len = 1000
    logged_prompt = prompt[:prompt_log_max_len] + ('...' if len(prompt) > prompt_log_max_len else '')
    logger.debug(f"LLM Prompt:\n--- START PROMPT ---\n{logged_prompt}\n--- END PROMPT ---")
    # --- >>> END ENHANCED LOGGING <<< ---

    try:
        if llm_provider == 'google':
            logger.debug("Sending request to Google API...")
            response = llm_client.generate_content(prompt)
            # --- >>> ADD GOOGLE RESPONSE LOGGING <<< ---
            try:
                raw_response_str = str(response) # Basic string conversion
                logger.debug(f"Raw Google Response (String): {raw_response_str[:500]}...") # Log truncated string
            except Exception as log_e:
                logger.warning(f"Could not log raw Google response details: {log_e}")

            if response and hasattr(response, 'text') and response.text:
                 logger.info("LLM call successful (Google).")
                 return response.text
            else:
                 failure_reason = "Unknown reason."
                 if not response: failure_reason = "API returned None response."
                 elif not hasattr(response, 'text'): failure_reason = "Response object lacks 'text' attribute."
                 elif not response.text: failure_reason = "Response text is empty."
                 elif hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
                      failure_reason = f"Blocked by API. Reason: {response.prompt_feedback.block_reason}"
                 elif hasattr(response, 'candidates') and response.candidates and getattr(response.candidates[0], 'finish_reason', None) != 'STOP':
                      failure_reason = f"Candidate finish reason: {response.candidates[0].finish_reason}"

                 logger.error(f"LLM call failed or returned empty/unexpected response (Google). Reason: {failure_reason}")
                 return None
            # --- >>> END LOGGING <<< ---

        elif llm_provider == 'openrouter':
            logger.debug("Sending request to OpenRouter API (via OpenAI client)...")
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant processing emergency incident data."},
                    {"role": "user", "content": prompt}
                ],
            )
            # --- >>> ADD OPENROUTER RESPONSE LOGGING <<< ---
            try:
                response_dict = response.model_dump() # Pydantic v2 method
                if response_dict.get('choices'):
                     for choice in response_dict['choices']:
                         if choice.get('message', {}).get('content'):
                             choice['message']['content'] = choice['message']['content'][:200] + '...'
                logger.debug(f"Raw OpenRouter Response (Dict): {json.dumps(response_dict, indent=2)}")
            except Exception as log_e:
                logger.warning(f"Could not log raw OpenRouter response details: {log_e}")

            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                content = response.choices[0].message.content
                logger.info("LLM call successful (OpenRouter).")
                return content.strip()
            else:
                 failure_reason = "Unknown reason."
                 if not response: failure_reason = "API returned None response."
                 elif not response.choices: failure_reason = "Response object has empty 'choices' list."
                 elif not response.choices[0].message: failure_reason = "First choice lacks 'message' attribute."
                 elif not response.choices[0].message.content: failure_reason = "First choice message content is empty."
                 finish_reason = response.choices[0].finish_reason if response and response.choices else "N/A"
                 logger.error(f"LLM call failed or returned empty choices (OpenRouter). Reason: {failure_reason}. Finish Reason: {finish_reason}")
                 return None
            # --- >>> END LOGGING <<< ---

    except Exception as e:
        logger.error(f"Error calling LLM API ({llm_provider} - {model_name}): {type(e).__name__}: {e}", exc_info=True)
        if hasattr(e, 'response'): # OpenAI specific
            try:
                err_details = e.response.json()
                logger.error(f"API Response Error Details: {json.dumps(err_details)}")
            except: pass
        elif hasattr(e, 'message'): # More generic
             logger.error(f"Error Message: {e.message}")
        return None

    logger.error(f"LLM call failed unexpectedly after API call for provider {llm_provider}.")
    return None


def summarize_incident(history: str, core_data: ReportCoreData) -> Optional[str]:
    """
    Generates an updated incident summary using the LLM.
    """
    prompt = f"""You are an AI assistant for emergency response. Your task is to synthesize incident information.
Given the following incident history (if any):
--- HISTORY START ---
{history if history else "No previous history available."}
--- HISTORY END ---

And the latest update information received:
--- LATEST UPDATE START ---
Timestamp: {core_data.timestamp.isoformat(timespec='seconds').replace('+00:00', 'Z')}
Incident Type: {core_data.incident_type or 'Not specified'}
Source System/Agency: {core_data.source or 'Unknown'}
Location Address: {core_data.location_address or 'Not specified'}
Location Coordinates: {f'({core_data.coordinates[0]:.5f}, {core_data.coordinates[1]:.5f})' if core_data.coordinates else 'Not specified'}
Description/Notes from this update: {core_data.description or 'No specific description in this update.'}
--- LATEST UPDATE END ---

Generate a concise, factual, and updated summary of the *current overall state* of the incident based on *all* available information (history and latest update).
Focus on the most critical and current details. If the history is empty, provide an initial summary based solely on the latest update.
Output only the summary text.
"""
    logger.debug(f"Calling LLM for incident summary (Report ID: {core_data.report_id[:8]})")
    return _call_llm(prompt)


def recommend_actions(summary: str, core_data: ReportCoreData) -> Optional[List[str]]:
    """
    Generates recommended actions based on the incident summary and latest data.
    """
    prompt = f"""You are an AI assistant advising emergency dispatchers or responders.
Based on the following incident summary and the latest update details:

--- CURRENT SUMMARY ---
{summary}
--- END SUMMARY ---

--- LATEST UPDATE DETAILS ---
Timestamp: {core_data.timestamp.isoformat(timespec='seconds').replace('+00:00', 'Z')}
Incident Type: {core_data.incident_type or 'Not specified'}
Source System/Agency: {core_data.source or 'Unknown'}
Location Address: {core_data.location_address or 'Not specified'}
Location Coordinates: {f'({core_data.coordinates[0]:.5f}, {core_data.coordinates[1]:.5f})' if core_data.coordinates else 'Not specified'}
Description/Notes from this update: {core_data.description or 'No specific description in this update.'}
--- END LATEST UPDATE ---

Suggest a short list of the most important next actions or considerations for responders or dispatchers.
Format the output as a simple list, with each action on a new line starting with '- '.
Example:
- Dispatch nearest appropriate unit.
- Advise caller to stay on the line if safe.
- Notify supervisor of incident escalation.

Output only the list of actions.
"""
    logger.debug(f"Calling LLM for recommended actions (Report ID: {core_data.report_id[:8]})")
    response_text = _call_llm(prompt)
    if response_text:
        actions = [line.strip('- ').strip() for line in response_text.splitlines() if line.strip() and line.strip().startswith('-')]
        if not actions:
             logger.warning(f"LLM response for actions received, but no lines started with '- '. Response: '{response_text[:100]}...'")
             return [f"AI Response (Format Issue?): {response_text}"]
        logger.debug(f"Successfully parsed {len(actions)} actions from LLM response.")
        return actions
    else:
        logger.warning(f"LLM call for actions returned None or empty string.")
        return None