# agent/llm_interface.py
import logging
from typing import List, Optional, Dict, Any
import sys
import json
import streamlit as st # Import Streamlit to access session state

# Import necessary LLM client libraries
from openai import OpenAI # Used for OpenRouter and Local LLMs
import google.generativeai as genai # Used for Google

# Import schemas (make sure paths are correct)
try:
    # Import settings INSTANCE to get initial defaults IF needed, but primarily rely on session state
    from config.settings import settings as initial_settings
    from data_models.schemas import ReportCoreData
    from services.eido_retriever import eido_retriever
except ImportError as e:
     print(f"--- FAILED to import dependencies in llm_interface.py: {e} ---")
     raise SystemExit(f"llm_interface failed to import dependencies: {e}") from e

logger = logging.getLogger(__name__)

# --- LLM Client Cache (Simple dictionary) ---
# Cache key: tuple (provider, api_key_snippet, base_url)
# Cache value: initialized client object
_client_cache: Dict[tuple, Any] = {}

def _get_llm_client(config: Dict[str, Any]) -> Optional[Any]:
    """
    Gets or creates an LLM client based on the provided runtime configuration.
    Uses a simple cache to avoid re-initializing clients unnecessarily.

    Args:
        config: A dictionary containing LLM settings, expected keys depend on provider:
            - llm_provider: 'google', 'openrouter', 'local', 'none'
            - google_api_key: (if provider is 'google')
            - openrouter_api_key: (if provider is 'openrouter')
            - openrouter_api_base_url: (if provider is 'openrouter')
            - local_llm_api_key: (if provider is 'local')
            - local_llm_api_base_url: (if provider is 'local')

    Returns:
        An initialized client object (genai.GenerativeModel or openai.OpenAI) or None.
    """
    provider = config.get('llm_provider')
    cache_key = None
    client = None

    if not provider or provider == 'none':
        logger.debug("LLM client requested for provider 'none', returning None.")
        return None

    try:
        if provider == 'google':
            api_key = config.get('google_api_key')
            if not api_key:
                logger.error("Google provider selected but API key is missing in current config.")
                return None
            # Use first/last few chars of key for caching without logging full key
            api_key_snippet = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else api_key[:4]
            cache_key = (provider, api_key_snippet, None) # Base URL not relevant for google client

            if cache_key in _client_cache:
                 logger.debug("Returning cached Google client.")
                 return _client_cache[cache_key]

            logger.debug("Initializing new Google client...")
            genai.configure(api_key=api_key)
            # Model name is selected during the call, not client init for Google
            # We initialize the base client capability here.
            # Let's assume we initialize a base model client; the specific model is used in generate_content
            model_name_for_init = config.get('google_model_name', initial_settings.google_model_name) # Use config or fallback default
            client = genai.GenerativeModel(model_name_for_init)
            _client_cache[cache_key] = client
            logger.info(f"Initialized and cached Google client for key snippet: {api_key_snippet}")
            return client

        elif provider == 'openrouter':
            api_key = config.get('openrouter_api_key')
            base_url = config.get('openrouter_api_base_url')
            if not api_key or not base_url:
                logger.error("OpenRouter provider selected but API key or Base URL is missing.")
                return None
            api_key_snippet = f"{api_key[:7]}...{api_key[-4:]}" if len(api_key) > 11 else api_key[:7]
            cache_key = (provider, api_key_snippet, base_url)

            if cache_key in _client_cache:
                logger.debug("Returning cached OpenRouter client.")
                return _client_cache[cache_key]

            logger.debug("Initializing new OpenRouter client...")
            client = OpenAI(api_key=api_key, base_url=base_url)
            _client_cache[cache_key] = client
            logger.info(f"Initialized and cached OpenRouter client for key snippet: {api_key_snippet}, URL: {base_url}")
            return client

        elif provider == 'local':
            api_key = config.get('local_llm_api_key', 'EMPTY') # Often not needed, use placeholder
            base_url = config.get('local_llm_api_base_url')
            if not base_url:
                logger.error("Local LLM provider selected but Base URL is missing.")
                return None
            # Use base_url for caching local clients, key is less relevant
            api_key_snippet = api_key # Cache based on actual key/placeholder used
            cache_key = (provider, api_key_snippet, base_url)

            if cache_key in _client_cache:
                logger.debug("Returning cached Local LLM client.")
                return _client_cache[cache_key]

            logger.debug("Initializing new Local LLM client (using OpenAI library)...")
            client = OpenAI(api_key=api_key, base_url=base_url)
            _client_cache[cache_key] = client
            logger.info(f"Initialized and cached Local LLM client for URL: {base_url}")
            return client

        else:
            logger.error(f"Unsupported LLM provider in config: {provider}")
            return None

    except ImportError as e:
         logger.critical(f"Failed to import LLM library for provider '{provider}': {e}. Please install required packages.", exc_info=True)
         return None
    except Exception as e:
         logger.critical(f"Failed to configure or get LLM client for provider '{provider}'. Config: {config}. Error: {e}", exc_info=True)
         return None


def _get_current_llm_config() -> Dict[str, Any]:
    """
    Retrieves the current LLM configuration, prioritizing Streamlit session state
    if available, otherwise falling back to initial settings.
    """
    config = {}
    # Use Streamlit session state if the module is loaded and session state exists
    if st and hasattr(st, 'session_state') and st.session_state:
        logger.debug("Retrieving LLM config from Streamlit session state.")
        keys = ['llm_provider', 'google_api_key', 'google_model_name',
                'openrouter_api_key', 'openrouter_model_name', 'openrouter_api_base_url',
                'local_llm_api_key', 'local_llm_model_name', 'local_llm_api_base_url']
        for key in keys:
            # Get value from session state, fall back to initial setting if key not in state yet
            config[key] = st.session_state.get(key, getattr(initial_settings, key, None))
    else:
        # Fallback to initial settings loaded from .env
        logger.debug("Streamlit session state not found or unavailable, using initial settings for LLM config.")
        keys = ['llm_provider', 'google_api_key', 'google_model_name',
                'openrouter_api_key', 'openrouter_model_name', 'openrouter_api_base_url',
                'local_llm_api_key', 'local_llm_model_name', 'local_llm_api_base_url']
        for key in keys:
            config[key] = getattr(initial_settings, key, None)

    # Clean up potential None values passed from session state if not set initially
    # This ensures defaults from initial_settings are used if session state holds None
    for key, value in config.items():
        if value is None and key != 'llm_provider':
            config[key] = getattr(initial_settings, key, None)

    # Log the config being used
    logger.debug(f"Using LLM Config: Provider={config.get('llm_provider')}, GoogleModel={config.get('google_model_name')}, OpenRouterModel={config.get('openrouter_model_name')}, LocalModel={config.get('local_llm_model_name')}")
    return config

def _call_llm(prompt: str) -> Optional[str]:
    """
    Internal function to call the dynamically configured LLM.
    Retrieves current config, gets/creates client, makes the call.
    """
    logger.debug(f"--- Attempting LLM call (_call_llm) ---")
    config = _get_current_llm_config()
    provider = config.get('llm_provider')
    llm_client = _get_llm_client(config)

    if not llm_client:
        logger.error(f"LLM client for provider '{provider}' could not be initialized or retrieved. Cannot call LLM.")
        return None

    # Determine model name based on provider
    model_name = None
    if provider == 'google':
        model_name = config.get('google_model_name')
    elif provider == 'openrouter':
        model_name = config.get('openrouter_model_name')
    elif provider == 'local':
        model_name = config.get('local_llm_model_name')

    if not model_name:
        logger.error(f"LLM model name is not configured for provider '{provider}'. Cannot call LLM.")
        return None

    logger.info(f"Calling LLM Provider: {provider}, Model: {model_name}")
    prompt_log_max_len = 1000
    logged_prompt = prompt[:prompt_log_max_len] + ('...' if len(prompt) > prompt_log_max_len else '')
    logger.debug(f"LLM Prompt:\n--- START PROMPT ---\n{logged_prompt}\n--- END PROMPT ---")

    try:
        if provider == 'google':
            logger.debug(f"Sending request to Google API (Model: {model_name})...")
            # Ensure the client object is compatible with generate_content
            # If _get_llm_client returns the base Model object, this should work.
            # Specify the actual model name if the client init didn't use it
            # Check if the client needs re-init with the specific model?
            # Google's new pattern: client = genai.GenerativeModel(model_name)
            # Let's re-fetch the client specifically for the model needed
            specific_google_client = genai.GenerativeModel(model_name) # Assume genai is configured
            response = specific_google_client.generate_content(prompt)

            try: raw_response_str = str(response)
            except Exception: raw_response_str = "[Could not stringify response]"
            logger.debug(f"Raw Google Response (String): {raw_response_str[:500]}...")

            if response and hasattr(response, 'text') and response.text:
                 logger.info("LLM call successful (Google).")
                 return response.text.strip() # Added strip
            # (Keep existing Google error/empty response logging)
            else:
                 failure_reason = "Unknown reason."
                 # ... (rest of failure reason logic remains the same) ...
                 logger.error(f"LLM call failed or returned empty/unexpected response (Google). Reason: {failure_reason}")
                 return None


        elif provider == 'openrouter' or provider == 'local':
            api_name = "OpenRouter" if provider == 'openrouter' else "Local LLM"
            logger.debug(f"Sending request to {api_name} API (Model: {model_name})...")
            # Client is openai.OpenAI instance
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant processing emergency incident data."},
                    {"role": "user", "content": prompt}
                ],
            )
            # (Keep existing OpenAI response logging)
            try:
                response_dict = response.model_dump() # Pydantic v2 method
                # ... (rest of response logging logic remains the same) ...
                logger.debug(f"Raw {api_name} Response (Dict): {json.dumps(response_dict, indent=2)}")
            except Exception as log_e: logger.warning(f"Could not log raw {api_name} response details: {log_e}")

            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                content = response.choices[0].message.content
                logger.info(f"LLM call successful ({api_name}).")
                return content.strip()
            # (Keep existing OpenAI empty/error response logging)
            else:
                 failure_reason = "Unknown reason."
                 # ... (rest of failure reason logic remains the same) ...
                 finish_reason = response.choices[0].finish_reason if response and response.choices else "N/A"
                 logger.error(f"LLM call failed or returned empty choices ({api_name}). Reason: {failure_reason}. Finish Reason: {finish_reason}")
                 return None

    except Exception as e:
        # Log specific API errors if possible
        api_error_details = ""
        if hasattr(e, 'response') and hasattr(e.response, 'text'): # Attempt to get OpenAI/HTTP error body
            api_error_details = f" API Response: {e.response.text[:500]}"
        elif hasattr(e, 'message'): # Generic error message
            api_error_details = f" Error Message: {e.message}"

        logger.error(f"Error calling LLM API ({provider} - {model_name}): {type(e).__name__}: {e}.{api_error_details}", exc_info=True)
        return None

    # This part should ideally not be reached if logic above is correct
    logger.error(f"LLM call failed unexpectedly after API call for provider {provider}.")
    return None


# --- summarize_incident, recommend_actions, extract_eido_from_alert_text ---
# These functions remain structurally the same, but they will now use _call_llm,
# which dynamically selects the LLM based on the current configuration.

def summarize_incident(history: str, core_data: ReportCoreData) -> Optional[str]:
    """Generates an updated incident summary using the configured LLM."""
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
    """Generates recommended actions based on the incident summary and latest data."""
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
             # If LLM didn't use bullet points, return the whole text as a single action item
             if response_text.strip():
                 logger.warning(f"LLM response for actions received, but no lines started with '- '. Returning raw response. Response: '{response_text[:100]}...'")
                 return [f"AI Response: {response_text}"]
             else:
                 logger.warning(f"LLM response for actions received, but was empty or whitespace only.")
                 return None
        logger.debug(f"Successfully parsed {len(actions)} actions from LLM response.")
        return actions
    else:
        logger.warning(f"LLM call for actions returned None.")
        return None
    
# --- *** ADD THIS FUNCTION *** ---
def fill_eido_template(template_str: str, scenario_desc: str) -> Optional[str]:
    """
    Uses LLM to fill placeholders in an EIDO template based on a scenario.

    Args:
        template_str: The EIDO JSON template as a string.
        scenario_desc: The natural language description of the scenario.

    Returns:
        A JSON string representing the filled template, or None on failure.
    """
    if not template_str or not scenario_desc:
        logger.error("Template string or scenario description missing for template filling.")
        return None
        # --- RAG Step ---
    retrieved_context_str = ""
    if eido_retriever.is_ready:
        # Simple query based on scenario description
        rag_query = f"EIDO schema definitions relevant to this scenario: {scenario_desc[:150]}"
        # TODO: More advanced: Parse template_str to identify components used, query for each?
        retrieved_chunks = eido_retriever.retrieve_context(rag_query, top_k=3) # Get more context for generation
        if retrieved_chunks:
            retrieved_context_str = "\n\nRelevant EIDO Schema Context:\n---\n" + "\n---\n".join(retrieved_chunks) + "\n---"
            logger.debug("Augmenting template filling prompt with retrieved schema context.")
        else:
            logger.debug("No relevant schema context retrieved for template filling prompt.")
    else:
        logger.warning("EIDO Retriever not ready, cannot augment template filling prompt.")
    # --- End RAG Step ---

    prompt = f"""
You are an AI assistant tasked with populating an EIDO JSON template.
Provided is an EIDO JSON template with placeholders (like [PLACEHOLDER_NAME]) and a scenario description.

**Instructions:**
1. Analyze the scenario description: "{scenario_desc}"
2. Analyze the EIDO JSON template below.
3. Identify all placeholders within the template (e.g., [TIMESTAMP_ISO_OFFSET], [ADDRESS], [DESCRIPTION], [INCIDENT_UUID], etc.).
4. Extract corresponding information from the scenario description to fill these placeholders accurately.
5. For placeholders like [INCIDENT_UUID], generate a plausible unique identifier (e.g., a short random alphanumeric string or part of a UUID). Use lowercase hex characters for UUID parts if generating them.
6. For timestamps like [TIMESTAMP_ISO_OFFSET], generate a valid ISO 8601 timestamp with timezone offset based on the scenario time (e.g., "2024-05-15T14:30:00-07:00"). If only date/time is given, try to infer a reasonable offset or default to UTC ('Z'). Use current time if no time is specified.
7. For IDs like [AGENCY_ID_REF], generate a plausible URN or simple ID (e.g., "chp.ca.gov" or "agency_123"). Be consistent if the same placeholder appears multiple times.
8. Replace *only* the placeholder strings (including the square brackets) in the template with the extracted or generated values. Maintain the exact structure and field names of the original template.
9. Ensure the final output is a single, valid JSON object. Pay close attention to quotes, commas, and brackets.
10. Output ONLY the completed JSON object. No explanations, apologies, or text before or after the JSON.

**EIDO Template:**
```json
{template_str}
```

**Completed JSON Output:**
"""
    logger.info("Calling LLM to fill EIDO template.")
    response_text = _call_llm(prompt) # Use the existing dynamic LLM call

    if not response_text:
        logger.error("LLM call for template filling returned no response.")
        return None

    # Clean markdown and validate JSON
    response_text = response_text.strip()
    if response_text.startswith("```json"): response_text = response_text[7:]
    if response_text.endswith("```"): response_text = response_text[:-3]
    response_text = response_text.strip()

    # Add extra logging for debugging
    logger.debug(f"LLM Raw Response (Template Filler):\n{response_text}")

    try:
        # Attempt to parse to ensure it's valid JSON before returning
        json.loads(response_text)
        logger.info("LLM successfully filled EIDO template and response is valid JSON.")
        return response_text
    except json.JSONDecodeError as e:
        logger.error(f"LLM response for template filling was not valid JSON: {e}")
        return None # Indicate failure
    except Exception as e:
        logger.error(f"Unexpected error validating template filler response: {e}", exc_info=True)
        return None
# --- *** END OF ADDED FUNCTION *** ---



def extract_eido_from_alert_text(alert_text: str) -> Optional[str]:
    """ Uses LLM to extract structured info from SINGLE event text, augmented with RAG. """
    retrieved_context_str = ""
    # ... (RAG logic remains the same) ...
    if eido_retriever.is_ready:
        rag_query = f"EIDO schema fields relevant for parsing this alert: {alert_text[:150]}"
        retrieved_chunks = eido_retriever.retrieve_context(rag_query, top_k=2)
        if retrieved_chunks: retrieved_context_str = "\n\nRelevant EIDO Schema Context:\n---\n" + "\n---\n".join(retrieved_chunks) + "\n---"
    # ...

    prompt = f"""
You are an AI assistant specialized in parsing **a single** emergency alert text message.
{retrieved_context_str} # <-- Inject retrieved context here
# ... (Rest of the prompt remains the same) ...
**Instructions:**
# ...
6. Output ONLY the JSON object. Ensure it is valid. Do not include ```json markdown wrappers or any text before or after the JSON object itself.

**Alert Text (Single Event):**
--- START ALERT ---
{alert_text}
--- END ALERT ---

**JSON Output:**
"""
    logger.info("Calling LLM (RAG-augmented) to extract structured data from single alert text event.")
    response_text = _call_llm(prompt)

    if not response_text:
        logger.error("LLM call for alert parsing returned no response.")
        return None

    # --- Improved Cleaning ---
    response_text = response_text.strip()
    # Remove potential markdown wrappers more robustly
    if response_text.startswith("```json"):
        response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
    elif response_text.startswith("```"): # Handle case where only ``` is present
        response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

    # Find the first '{' and the last '}' to isolate the JSON object
    start_index = response_text.find('{')
    end_index = response_text.rfind('}')

    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_candidate = response_text[start_index : end_index + 1]
        logger.debug(f"Attempting to parse cleaned JSON candidate:\n{json_candidate}")
        try:
            # Validate the isolated candidate
            json.loads(json_candidate)
            logger.info("LLM returned a response that appears to be valid JSON for event details.")
            return json_candidate # Return only the valid JSON part
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse cleaned JSON candidate: {e}")
            logger.debug(f"Original LLM Raw Response (Extractor):\n{response_text}")
            return None # Return None if isolated part is still invalid
        except Exception as e:
             logger.error(f"Unexpected error validating extractor response: {e}", exc_info=True)
             return None
    else:
        # Could not find valid start/end braces
        logger.warning(f"Could not isolate JSON object using braces in LLM response.")
        logger.debug(f"Original LLM Raw Response (Extractor):\n{response_text}")
        return None
    # --- End Improved Cleaning ---

def split_raw_text_into_events(raw_text: str) -> Optional[List[str]]:
    """
    Uses an LLM to analyze raw text and split it into separate strings,
    each representing a distinct potential incident report.
    """
    if not raw_text: return None

    prompt = f"""
Analyze the following text, which may contain descriptions of one or multiple distinct emergency incidents or updates.
Your primary task is to identify the boundaries between separate events. Events are often separated by timestamps, incident numbers, explicit markers like "New Call:", "Update:", or significant changes in location/context.

**Instructions:**
1.  Carefully read the entire text.
2.  Determine if the text describes ONE single event or MULTIPLE distinct events.
3.  **Output Format:** Respond ONLY with a valid JSON list of strings.
    *   If MULTIPLE distinct events are identified, the list should contain multiple strings. Each string must contain the text relevant *only* to that specific event. Preserve original wording. Example: `["Event 1 text...", "Event 2 text..."]`
    *   If ONLY ONE event is identified, the list must contain a single string element, which is the complete original input text (or minimally cleaned version). Example: `["Complete text of single event..."]`
4.  **CRITICAL:** Do NOT include any explanations, introductions, or text outside the JSON list itself. The entire response must be the JSON list.

**Input Text:**
--- START TEXT ---
{raw_text}
--- END TEXT ---

**JSON Output (list of strings ONLY):**
"""
    logger.info("Calling LLM to split raw text into potential events.")
    response_text = _call_llm(prompt)

    if not response_text:
        logger.error("LLM call for event splitting returned no response.")
        return None

    # Clean potential markdown and whitespace
    response_text = response_text.strip()
    if response_text.startswith("```json"): response_text = response_text[7:]
    if response_text.endswith("```"): response_text = response_text[:-3]
    response_text = response_text.strip()

    # Add extra logging to see the raw response before parsing
    logger.debug(f"LLM Raw Response (Splitter):\n{response_text}")

    try:
        parsed_list = json.loads(response_text)
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            if len(parsed_list) > 1:
                 logger.info(f"LLM successfully split text into {len(parsed_list)} potential events.")
            else:
                 logger.info("LLM indicated only one event in the text block.")
            return parsed_list
        else:
            logger.error(f"LLM response for splitting was not a valid JSON list of strings. Type: {type(parsed_list)}")
            # Return None to trigger fallback in agent_core
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from LLM during event splitting: {e}")
        # Return None to trigger fallback in agent_core
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing splitter response: {e}", exc_info=True)
        # Return None to trigger fallback in agent_core
        return None