# agent/llm_interface.py
import logging
from typing import List, Optional, Dict, Any, Tuple
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
            # Model name is part of client init for Google's new SDK structure
            model_name_for_init = config.get('google_model_name', initial_settings.google_model_name)
            cache_key = (provider, api_key_snippet, model_name_for_init)

            if cache_key in _client_cache:
                 logger.debug(f"Returning cached Google client for model {model_name_for_init}.")
                 return _client_cache[cache_key]

            logger.debug(f"Initializing new Google client for model {model_name_for_init}...")
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(model_name_for_init)
            _client_cache[cache_key] = client
            logger.info(f"Initialized and cached Google client for model {model_name_for_init}, key snippet: {api_key_snippet}")
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
            api_key = config.get('local_llm_api_key', 'EMPTY')
            base_url = config.get('local_llm_api_base_url')
            if not base_url:
                logger.error("Local LLM provider selected but Base URL is missing.")
                return None
            api_key_snippet = api_key
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
         # Log sensitive info carefully
         safe_config = {k: v for k, v in config.items() if 'key' not in k.lower()}
         logger.critical(f"Failed to configure or get LLM client for provider '{provider}'. Config (sanitized): {json.dumps(safe_config, default=str)}. Error: {e}", exc_info=True)
         return None


def _get_current_llm_config() -> Dict[str, Any]:
    """
    Retrieves the current LLM configuration, prioritizing Streamlit session state
    if available, otherwise falling back to initial settings.
    """
    config = {}
    if 'st' in sys.modules and hasattr(st, 'session_state') and st.session_state: # Check if Streamlit context exists
        logger.debug("Retrieving LLM config from Streamlit session state.")
        keys_to_sync = [
            'llm_provider', 'google_api_key', 'google_model_name',
            'openrouter_api_key', 'openrouter_model_name', 'openrouter_api_base_url',
            'local_llm_api_key', 'local_llm_model_name', 'local_llm_api_base_url'
        ]
        for key in keys_to_sync:
            # Ensure session state doesn't override with None if initial setting exists
            session_val = st.session_state.get(key)
            initial_val = getattr(initial_settings, key, None)
            config[key] = session_val if session_val is not None else initial_val
    else:
        logger.debug("Streamlit session state not found or unavailable, using initial settings for LLM config.")
        config = initial_settings.model_dump() # Get all initial settings

    # Log the config being used (excluding full API keys for security)
    log_config = {k: (v if 'api_key' not in k.lower() else (f"{v[:4]}..." if v else "None")) for k,v in config.items()}
    logger.debug(f"Using LLM Config: {json.dumps(log_config, default=str)}")
    return config

def _call_llm(prompt: str, is_json_output: bool = False) -> Optional[str]:
    """
    Internal function to call the dynamically configured LLM.
    is_json_output: If True, configures Google Gemini for JSON output.
    """
    logger.debug(f"--- Attempting LLM call (_call_llm) --- JSON Mode: {is_json_output}")
    config = _get_current_llm_config()
    provider = config.get('llm_provider')

    # For Google, model name is tied to the client instance now.
    # For others, model name is passed in the API call.
    # So, _get_llm_client needs the model name for Google.
    llm_client = _get_llm_client(config)

    if not llm_client:
        logger.error(f"LLM client for provider '{provider}' could not be initialized. Cannot call LLM.")
        return None

    model_name = None # This will be used by OpenAI-compatible clients
    if provider == 'openrouter': model_name = config.get('openrouter_model_name')
    elif provider == 'local': model_name = config.get('local_llm_model_name')
    # For Google, the model is already part of `llm_client` from `_get_llm_client`

    actual_model_being_used = model_name if provider in ['openrouter', 'local'] else config.get('google_model_name')

    if (provider == 'openrouter' or provider == 'local') and not actual_model_being_used:
        logger.error(f"LLM model name is not configured for provider '{provider}'. Cannot call LLM.")
        return None
    elif provider == 'google' and not actual_model_being_used:
         logger.error(f"LLM model name is not configured for provider '{provider}'. Cannot call LLM.")
         return None


    logger.info(f"Calling LLM Provider: {provider}, Model: {actual_model_being_used}")
    prompt_log_max_len = 1000
    logged_prompt = prompt[:prompt_log_max_len] + ('...' if len(prompt) > prompt_log_max_len else '')
    # logger.debug(f"LLM Prompt:\n--- START PROMPT ---\n{logged_prompt}\n--- END PROMPT ---") # DEBUG level can be verbose

    try:
        if provider == 'google':
            logger.debug(f"Sending request to Google API (Model: {actual_model_being_used}). JSON mode: {is_json_output}")
            generation_config = None
            if is_json_output:
                generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

            # llm_client should be the correctly initialized GenerativeModel instance
            response = llm_client.generate_content(prompt, generation_config=generation_config)

            raw_response_str = "[Could not stringify response]"
            try: raw_response_str = str(response)
            except Exception: pass
            # logger.debug(f"Raw Google Response (String): {raw_response_str[:500]}...") # DEBUG

            if response and hasattr(response, 'text') and response.text:
                 logger.info("LLM call successful (Google).")
                 return response.text.strip()
            else:
                 failure_reason = "Unknown reason."
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     failure_reason = f"Prompt Feedback: {response.prompt_feedback}"
                 elif not response or not hasattr(response, 'text') or not response.text:
                     failure_reason = "Response object was empty or missing text part."
                 logger.error(f"LLM call failed or returned empty/unexpected response (Google). Reason: {failure_reason}")
                 # Log candidate parts if available for debugging
                 if response and hasattr(response, 'candidates'): logger.error(f"Google Response Candidates: {response.candidates}")
                 return None

        elif provider == 'openrouter' or provider == 'local':
            api_name = "OpenRouter" if provider == 'openrouter' else "Local LLM"
            logger.debug(f"Sending request to {api_name} API (Model: {actual_model_being_used}). JSON mode: {is_json_output}")

            messages=[
                {"role": "system", "content": "You are an AI assistant processing emergency incident data. If the user asks for JSON output, ensure your response is ONLY the valid JSON object."},
                {"role": "user", "content": prompt}
            ]

            request_params = {"model": actual_model_being_used, "messages": messages}
            # Attempt to use JSON mode if requested and likely supported
            if is_json_output:
                 # Check if model name suggests OpenAI or Anthropic v3+, or Command R+ (more likely to support JSON mode)
                 if provider != 'local' or any(m in actual_model_being_used.lower() for m in ['gpt', 'claude-3', 'command-r']):
                     request_params["response_format"] = {"type": "json_object"}
                     logger.debug("Attempting to use JSON mode for OpenAI-compatible API.")
                 else:
                    logger.warning(f"JSON mode requested for local model '{actual_model_being_used}', but explicit JSON mode support is uncertain. Relying on prompt instructions.")

            response = llm_client.chat.completions.create(**request_params)

            try:
                response_dict = response.model_dump()
                # logger.debug(f"Raw {api_name} Response (Dict): {json.dumps(response_dict, indent=2)}") # DEBUG
            except Exception as log_e: logger.warning(f"Could not log raw {api_name} response details: {log_e}")

            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                content = response.choices[0].message.content
                logger.info(f"LLM call successful ({api_name}).")
                return content.strip()
            else:
                 failure_reason = "Unknown reason."
                 if not response or not response.choices: failure_reason = "Response or choices list was empty."
                 elif not response.choices[0].message: failure_reason = "First choice message was empty."
                 elif not response.choices[0].message.content: failure_reason = "Message content was empty."
                 finish_reason = response.choices[0].finish_reason if response and response.choices else "N/A"
                 logger.error(f"LLM call failed or returned empty choices ({api_name}). Reason: {failure_reason}. Finish Reason: {finish_reason}")
                 return None
    except Exception as e:
        api_error_details = ""
        if hasattr(e, 'response') and hasattr(e.response, 'text'): api_error_details = f" API Response: {e.response.text[:500]}"
        elif hasattr(e, 'message'): api_error_details = f" Error Message: {str(e)}" # Use str(e)
        logger.error(f"Error calling LLM API ({provider} - {actual_model_being_used}): {type(e).__name__}: {e}.{api_error_details}", exc_info=True)
        return None
    logger.error(f"LLM call failed unexpectedly after API call for provider {provider}.")
    return None


def summarize_incident(history: str, core_data: ReportCoreData) -> Optional[str]:
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
ZIP Code: {core_data.zip_code or 'Not specified'}
Description/Notes from this update: {core_data.description or 'No specific description in this update.'}
--- LATEST UPDATE END ---

Generate a concise, factual, and updated summary of the *current overall state* of the incident based on *all* available information (history and latest update).
Focus on the most critical and current details. If the history is empty, provide an initial summary based solely on the latest update.
Output only the summary text.
"""
    logger.debug(f"Calling LLM for incident summary (Report ID: {core_data.report_id[:8]})")
    return _call_llm(prompt)


def recommend_actions(summary: str, core_data: ReportCoreData) -> Optional[List[str]]:
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
ZIP Code: {core_data.zip_code or 'Not specified'}
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
        if not actions and response_text.strip():
             logger.warning(f"LLM response for actions received, but no lines started with '- '. Returning raw response. Response: '{response_text[:100]}...'")
             return [f"AI Response: {response_text}"]
        elif not actions:
             logger.warning(f"LLM response for actions received, but was empty or whitespace only.")
             return None
        logger.debug(f"Successfully parsed {len(actions)} actions from LLM response.")
        return actions
    logger.warning(f"LLM call for actions returned None.")
    return None


def fill_eido_template(template_str: str, scenario_desc: str) -> Optional[str]:
    """Uses LLM to fill placeholders in an EIDO template based on a scenario, augmented with RAG."""
    if not template_str or not scenario_desc:
        logger.error("Template string or scenario description missing for template filling.")
        return None

    retrieved_context_str = ""
    if eido_retriever.is_ready:
        rag_query = f"EIDO schema definitions and field explanations relevant to this scenario: {scenario_desc[:150]}"
        retrieved_chunks = eido_retriever.retrieve_context(rag_query, top_k=3)
        if retrieved_chunks:
            retrieved_context_str = "\n\n**Relevant EIDO Schema Context (Use this to ensure correct field names, types, and common values like URNs or registry texts):**\n---\n" + "\n---\n".join(retrieved_chunks) + "\n---"
            logger.debug("Augmenting EIDO template filling prompt with retrieved schema context.")
        else: logger.debug("No relevant schema context retrieved for EIDO template filling prompt.")
    else: logger.warning("EIDO Retriever not ready, cannot augment EIDO template filling prompt.")

    prompt = f"""
You are an AI assistant tasked with populating an EIDO JSON template.
Provided is an EIDO JSON template with placeholders (like [PLACEHOLDER_NAME] or {{PLACEHOLDER_NAME}}) and a scenario description.
{retrieved_context_str}

**Instructions:**
1.  Carefully analyze the scenario description: "{scenario_desc}"
2.  Analyze the EIDO JSON template below.
3.  Identify all placeholders within the template (e.g., [TIMESTAMP_ISO_OFFSET], [LOCATION_ADDRESS], [DESCRIPTION], [INCIDENT_UUID], [EXTERNAL_CAD_ID], [LATITUDE], [LONGITUDE], [ZIP_CODE], [SOURCE_AGENCY_ID], [SOURCE_AGENCY_NAME], etc.).
4.  Extract corresponding information from the scenario description to fill these placeholders accurately.
5.  For placeholders like [INCIDENT_UUID], [EIDO_MESSAGE_UUID], [LOCATION_UUID], [NOTE_UUID], [AUTHOR_UUID], [SOURCE_AGENCY_UUID], generate plausible unique identifiers. Use standard UUID format (e.g., "urn:uuid:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx") or simple unique strings like "inc-123ab", "note-xyz789" if the template implies simpler IDs. Ensure consistency if the same placeholder type (e.g. a location ID) is referenced multiple times.
6.  For timestamps like [TIMESTAMP_ISO_OFFSET], generate a valid ISO 8601 timestamp with timezone offset based on the scenario time (e.g., "2024-05-15T14:30:00-07:00"). If only date/time is given, try to infer a reasonable offset or default to UTC ('Z'). Use current time if no time is specified.
7.  For IDs like [SOURCE_AGENCY_ID], generate a plausible URN or simple ID (e.g., "urn:nena:agency:anytownpd" or "anytown.pd.gov"). Use the provided schema context if it suggests formats.
8.  Replace *only* the placeholder strings (including the square/curly brackets) in the template with the extracted or generated values. Maintain the exact structure, field names, and data types (string, number, boolean) of the original template.
9.  Ensure the final output is a single, valid JSON object. Pay close attention to quotes, commas, and brackets.
10. **CRITICAL: Output ONLY the completed JSON object. No explanations, apologies, or text before or after the JSON.**

**EIDO Template:**
```json
{template_str}
```

**Scenario Description for Filling:**
"{scenario_desc}"

**Completed JSON Output (JSON object only):**
"""
    logger.info("Calling LLM (RAG-augmented) to fill EIDO template.")
    response_text = _call_llm(prompt, is_json_output=True)

    if not response_text:
        logger.error("LLM call for template filling returned no response.")
        return None

    # Clean markdown and validate JSON
    response_text = response_text.strip()
    if response_text.startswith("```json"): response_text = response_text[7:]
    if response_text.endswith("```"): response_text = response_text[:-3]
    response_text = response_text.strip()

    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        response_text = response_text[start_brace : end_brace + 1]
    else:
        logger.error(f"Could not isolate a JSON object in the LLM response for template filling. Raw response: {response_text[:200]}...")
        return None


    # logger.debug(f"LLM Raw Response (Template Filler, after cleaning):\n{response_text}")
    try:
        json.loads(response_text) # Validate JSON
        logger.info("LLM successfully filled EIDO template and response is valid JSON.")
        return response_text
    except json.JSONDecodeError as e:
        logger.error(f"LLM response for template filling was not valid JSON: {e}. Response: {response_text[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error validating template filler response: {e}", exc_info=True)
        return None


def extract_eido_from_alert_text(alert_text: str) -> Optional[str]:
    """Uses LLM to extract structured info from SINGLE event text, augmented with RAG. Returns a JSON string."""
    retrieved_context_str = ""
    if eido_retriever.is_ready:
        rag_query = f"EIDO schema fields relevant for parsing this emergency alert: {alert_text[:150]}"
        retrieved_chunks = eido_retriever.retrieve_context(rag_query, top_k=3) # Get more context
        if retrieved_chunks:
            retrieved_context_str = "\n\n**Relevant EIDO Schema Context (Use this to guide field extraction and naming):**\n---\n" + "\n---\n".join(retrieved_chunks) + "\n---"
            logger.debug("Augmenting alert parsing prompt with retrieved schema context.")
        else: logger.debug("No relevant schema context retrieved for alert parsing prompt.")
    else: logger.warning("EIDO Retriever not ready, cannot augment alert parsing prompt.")

    prompt = f"""
You are an AI assistant specialized in parsing **a single** emergency alert text message into a structured JSON object.
{retrieved_context_str}

**Instructions:**
1.  Analyze the provided alert text.
2.  Extract the following key pieces of information if present:
    *   `incident_type`: (e.g., "Structure Fire", "Vehicle Collision", "Medical Emergency", "Traffic Stop", "Suspicious Person") - Be specific.
    *   `timestamp_iso`: The date and time of the event or report, converted to ISO 8601 format (e.g., "YYYY-MM-DDTHH:MM:SSZ" or "YYYY-MM-DDTHH:MM:SS+/-HH:MM"). If only time is given, assume today's date. Try to infer timezone if mentioned (PST, EST etc.), otherwise assume UTC or local if ambiguous.
    *   `location_address`: The street address or intersection (e.g., "123 Main St, Anytown", "Elm St / Oak Ave").
    *   `location_description`: Any additional descriptive location information (e.g., "Apt 5B", "behind the mall", "northbound lanes").
    *   `coordinates`: Latitude and Longitude as a list `[lat, lon]` if explicitly mentioned.
    *   `zip_code`: The postal or ZIP code if mentioned.
    *   `description`: A concise summary of what happened or was reported, including key details like number of vehicles, patient conditions, etc.
    *   `source_agency`: The reporting agency, unit, or system (e.g., "CHP Dispatch", "Engine 4", "CAD System X", "Caller").
    *   `external_id`: Any distinct incident number or call ID (e.g., "CAD2024-00123", "CHPLog#5678").
3.  If a piece of information is not present, omit the key or set its value to `null`.
4.  Format the output as a single JSON object.
5.  Pay close attention to data types: strings should be in double quotes, numbers should be numbers (not strings), lists for coordinates. Coordinates should be numbers (float or int).
6.  **CRITICAL: Output ONLY the JSON object. Ensure it is valid. Do not include ```json markdown wrappers or any text before or after the JSON object itself.**

**Alert Text (Single Event):**
--- START ALERT ---
{alert_text}
--- END ALERT ---

**JSON Output (JSON object only):**
"""
    logger.info("Calling LLM (RAG-augmented) to extract structured data from single alert text event.")
    response_text = _call_llm(prompt, is_json_output=True)

    if not response_text:
        logger.error("LLM call for alert parsing returned no response.")
        return None

    # Clean markdown and isolate JSON
    response_text = response_text.strip()
    if response_text.startswith("```json"): response_text = response_text[7:]
    if response_text.endswith("```"): response_text = response_text[:-3]
    response_text = response_text.strip()

    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        json_candidate = response_text[start_brace : end_brace + 1]
        # logger.debug(f"Attempting to parse cleaned JSON candidate for alert extraction:\n{json_candidate}")
        try:
            json.loads(json_candidate) # Validate
            logger.info("LLM returned a response that appears to be valid JSON for event details.")
            return json_candidate
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse cleaned JSON candidate from alert parser: {e}. Raw response: {response_text[:500]}...")
            return None
    else:
        logger.warning(f"Could not isolate JSON object in LLM response for alert parser. Raw response: {response_text[:200]}...")
        return None


def split_raw_text_into_events(raw_text: str) -> Optional[List[str]]:
    """
    Uses an LLM to analyze raw text and split it into separate strings,
    each representing a distinct potential incident report.
    """
    if not raw_text: return None

    prompt = f"""
Analyze the following text, which may contain descriptions of one or multiple distinct emergency incidents or updates.
Your primary task is to identify the boundaries between separate events. Events are often separated by timestamps, incident numbers, explicit markers like "New Call:", "Update:", "Unit:", "EVENT:", or significant changes in location/context, or double line breaks.

**Instructions:**
1.  Carefully read the entire text.
2.  Determine if the text describes ONE single event or MULTIPLE distinct events.
3.  **Output Format:** Respond ONLY with a valid JSON list of strings.
    *   If MULTIPLE distinct events are identified, the list should contain multiple strings. Each string must contain the text relevant *only* to that specific event. Preserve original wording. Example: `["Event 1 text, including its timestamp and ID...", "Event 2 text with its details..."]`
    *   If ONLY ONE event is identified, the list must contain a single string element, which is the complete original input text (or minimally cleaned version). Example: `["Complete text of single event..."]`
4.  **CRITICAL:** Do NOT include any explanations, introductions, or text outside the JSON list itself. The entire response must be the JSON list.

**Input Text:**
--- START TEXT ---
{raw_text}
--- END TEXT ---

**JSON Output (list of strings ONLY):**
"""
    logger.info("Calling LLM to split raw text into potential events.")
    response_text = _call_llm(prompt, is_json_output=True) # Expecting JSON list

    if not response_text:
        logger.error("LLM call for event splitting returned no response.")
        return None

    response_text = response_text.strip()
    if response_text.startswith("```json"): response_text = response_text[7:]
    if response_text.endswith("```"): response_text = response_text[:-3]
    response_text = response_text.strip()

    start_bracket = response_text.find('[')
    end_bracket = response_text.rfind(']')
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        response_text = response_text[start_bracket : end_bracket + 1]
    else:
        logger.error(f"Could not isolate a JSON list in the LLM response for splitting. Raw response: {response_text[:200]}...")
        return None # Fallback will treat as single event

    # logger.debug(f"LLM Raw Response (Splitter, after cleaning):\n{response_text}")
    try:
        parsed_list = json.loads(response_text)
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            if not parsed_list:
                logger.warning("LLM splitter returned an empty list. Treating as single event.")
                return None # Fallback
            # Filter out any empty strings that might be in the list
            cleaned_list = [item for item in parsed_list if item.strip()]
            if not cleaned_list:
                 logger.warning("LLM splitter returned a list with only empty strings. Treating as single event.")
                 return None # Fallback
            if len(cleaned_list) > 1: logger.info(f"LLM successfully split text into {len(cleaned_list)} non-empty potential events.")
            else: logger.info("LLM indicated only one event in the text block (returned list with 1 non-empty item).")
            return cleaned_list
        else:
            logger.error(f"LLM response for splitting was not a valid JSON list of strings. Type: {type(parsed_list)}. Content: {str(parsed_list)[:200]}")
            return None # Fallback
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from LLM during event splitting: {e}. Response: {response_text[:500]}")
        return None # Fallback
    except Exception as e:
        logger.error(f"Unexpected error parsing splitter response: {e}", exc_info=True)
        return None # Fallback

def geocode_address_with_llm(address_text: str) -> Optional[Tuple[float, float]]:
    """
    Uses an LLM to attempt to geocode an address string.
    Returns (latitude, longitude) or None.
    """
    if not address_text or not isinstance(address_text, str):
        logger.warning("LLM geocoding: Invalid address text provided.")
        return None

    config = _get_current_llm_config()
    if config.get('llm_provider') == 'none':
        logger.info("LLM geocoding skipped: LLM provider is 'none'.")
        return None

    prompt = f"""
You are an expert geocoding assistant. Your task is to determine the geographic coordinates (latitude and longitude) for the given address.

**Address to Geocode:**
"{address_text}"

**Instructions:**
1.  Analyze the address carefully.
2.  If you can confidently determine the latitude and longitude, provide them.
3.  **CRITICAL Output Format:** Respond ONLY with a single, valid JSON object in the following format:
    `{{"latitude": <float_value>, "longitude": <float_value>}}`
    Example: `{{"latitude": 34.0522, "longitude": -118.2437}}`
4.  If you cannot determine the coordinates with reasonable accuracy, or if the address is ambiguous or clearly invalid, respond ONLY with an empty JSON object: `{{}}`.
5.  Do NOT include any explanations, apologies, or text outside the JSON object itself.

**JSON Output:**
"""
    logger.info(f"Calling LLM for geocoding address: '{address_text[:100]}...'")
    response_text = _call_llm(prompt, is_json_output=True)

    if not response_text:
        logger.error(f"LLM geocoding failed: No response received for address '{address_text[:100]}...'")
        return None

    response_text = response_text.strip()
    # Additional cleaning for common LLM habits (like ```json ... ```)
    if response_text.startswith("```json"): response_text = response_text[7:]
    if response_text.endswith("```"): response_text = response_text[:-3]
    response_text = response_text.strip()

    try:
        data = json.loads(response_text)
        if not isinstance(data, dict):
            logger.warning(f"LLM geocoding: Response was not a JSON object for '{address_text[:100]}...'. Response: {response_text[:200]}")
            return None

        if not data: # Empty JSON object {}
            logger.info(f"LLM geocoding: LLM indicated it could not determine coordinates for '{address_text[:100]}...'.")
            return None

        lat = data.get("latitude")
        lon = data.get("longitude")

        if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
            # Basic validation for lat/lon ranges
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                logger.info(f"LLM geocoding successful for '{address_text[:100]}...': ({float(lat)}, {float(lon)})")
                return (float(lat), float(lon))
            else:
                logger.warning(f"LLM geocoding: Coordinates out of valid range for '{address_text[:100]}...'. Lat: {lat}, Lon: {lon}")
                return None
        else:
            logger.warning(f"LLM geocoding: Latitude or longitude missing, or not numbers, for '{address_text[:100]}...'. Data: {data}")
            return None
    except json.JSONDecodeError:
        logger.error(f"LLM geocoding: Failed to parse JSON response for '{address_text[:100]}...'. Response: {response_text[:200]}")
        return None
    except Exception as e:
        logger.error(f"LLM geocoding: Unexpected error processing LLM response for '{address_text[:100]}...': {e}", exc_info=True)
        return None