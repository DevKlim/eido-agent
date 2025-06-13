import logging
from typing import List, Optional, Dict, Any, Tuple
import sys
import os
import json
import streamlit as st 
import datetime
from openai import OpenAI 
import google.generativeai as genai 

try:
    from config.settings import settings as initial_settings
    from data_models.schemas import ReportCoreData
    from services.eido_retriever import eido_retriever
    from services.campus_geocoder import get_ucsd_coordinates
except ImportError as e:
     print(f"--- FAILED to import dependencies in llm_interface.py: {e} ---")
     raise SystemExit(f"llm_interface failed to import dependencies: {e}") from e

logger = logging.getLogger(__name__)

# --- Prompt Loading ---
def load_prompts_from_json() -> Dict[str, str]:
    """Loads all prompts from the JSON library file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "prompt_library.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        logger.info(f"Successfully loaded {len(prompts)} prompts from {json_path}")
        return prompts
    except FileNotFoundError:
        logger.critical(f"FATAL: Prompt library not found at {json_path}. The application cannot function without it.")
        raise
    except json.JSONDecodeError as e:
        logger.critical(f"FATAL: Failed to parse prompt library JSON at {json_path}: {e}")
        raise
    except Exception as e:
        logger.critical(f"FATAL: An unexpected error occurred while loading prompts: {e}")
        raise

# Load prompts into a global dictionary when the module is imported
PROMPTS = load_prompts_from_json()


_client_cache: Dict[tuple, Any] = {}

def _get_llm_client(config: Dict[str, Any]) -> Optional[Any]:
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
            api_key_snippet = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else api_key[:4]
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
         safe_config = {k: v for k, v in config.items() if 'key' not in k.lower()}
         logger.critical(f"Failed to configure or get LLM client for provider '{provider}'. Config (sanitized): {json.dumps(safe_config, default=str)}. Error: {e}", exc_info=True)
         return None


def _get_current_llm_config() -> Dict[str, Any]:
    config = {}
    if 'st' in sys.modules and hasattr(st, 'session_state') and st.session_state: 
        logger.debug("Retrieving LLM config from Streamlit session state.")
        keys_to_sync = [
            'llm_provider', 'google_api_key', 'google_model_name',
            'openrouter_api_key', 'openrouter_model_name', 'openrouter_api_base_url',
            'local_llm_api_key', 'local_llm_model_name', 'local_llm_api_base_url'
        ]
        for key in keys_to_sync:
            session_val = st.session_state.get(key)
            initial_val = getattr(initial_settings, key, None)
            config[key] = session_val if session_val is not None else initial_val
    else:
        logger.debug("Streamlit session state not found or unavailable, using initial settings for LLM config.")
        config = initial_settings.model_dump() 

    log_config = {k: (v if 'api_key' not in k.lower() else (f"{v[:4]}..." if v else "None")) for k,v in config.items()}
    logger.debug(f"Using LLM Config: {json.dumps(log_config, default=str)}")
    return config

def _call_llm(prompt: str, is_json_output: bool = False) -> Optional[str]:
    logger.debug(f"--- Attempting LLM call (_call_llm) --- JSON Mode: {is_json_output}")
    config = _get_current_llm_config()
    provider = config.get('llm_provider')
    llm_client = _get_llm_client(config)

    if not llm_client:
        logger.error(f"LLM client for provider '{provider}' could not be initialized. Cannot call LLM.")
        return None

    model_name = None 
    if provider == 'openrouter': model_name = config.get('openrouter_model_name')
    elif provider == 'local': model_name = config.get('local_llm_model_name')
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
    
    try:
        if provider == 'google':
            logger.debug(f"Sending request to Google API (Model: {actual_model_being_used}). JSON mode: {is_json_output}")
            generation_config = None
            if is_json_output:
                try:
                    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
                    logger.info(f"Using JSON response_mime_type for Google model {actual_model_being_used}.")
                except Exception as e_json_config:
                    logger.warning(f"Could not set JSON response_mime_type for {actual_model_being_used}: {e_json_config}. Proceeding without it.")
            
            response = llm_client.generate_content(prompt, generation_config=generation_config)
            
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
            if is_json_output:
                 if any(m_alias in actual_model_being_used.lower() for m_alias in ['gpt-3.5-turbo', 'gpt-4', 'claude-3', 'command-r', 'gemini']):
                     request_params["response_format"] = {"type": "json_object"}
                     logger.debug(f"Attempting to use OpenAI-compatible JSON mode for {api_name} model {actual_model_being_used}.")
                 else:
                    logger.warning(f"JSON mode requested for {api_name} model '{actual_model_being_used}', but explicit JSON mode support is uncertain. Relying on prompt instructions.")
            
            response = llm_client.chat.completions.create(**request_params)
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
        elif hasattr(e, 'message'): api_error_details = f" Error Message: {str(e)}" 
        logger.error(f"Error calling LLM API ({provider} - {actual_model_being_used}): {type(e).__name__}: {e}.{api_error_details}", exc_info=True)
        return None
    logger.error(f"LLM call failed unexpectedly after API call for provider {provider}.")
    return None


def summarize_incident(history: str, core_data: ReportCoreData) -> Optional[str]:
    prompt = PROMPTS['SUMMARIZE_INCIDENT'].format(
        history=history if history else "No previous history available.",
        timestamp=core_data.timestamp.isoformat(timespec='seconds').replace('+00:00', 'Z'),
        incident_type=core_data.incident_type or 'Not specified',
        source=core_data.source or 'Unknown',
        location_address=core_data.location_address or 'Not specified',
        coordinates=f'({core_data.coordinates[0]:.5f}, {core_data.coordinates[1]:.5f})' if core_data.coordinates else 'Not specified',
        zip_code=core_data.zip_code or 'Not specified',
        description=core_data.description or 'No specific description in this update.'
    )
    logger.debug(f"Calling LLM for incident summary (Report ID: {core_data.report_id[:8]})")
    return _call_llm(prompt)


def recommend_actions(summary: str, core_data: ReportCoreData) -> Optional[List[str]]:
    prompt = PROMPTS['RECOMMEND_ACTIONS'].format(
        summary=summary,
        timestamp=core_data.timestamp.isoformat(timespec='seconds').replace('+00:00', 'Z'),
        incident_type=core_data.incident_type or 'Not specified',
        source=core_data.source or 'Unknown',
        location_address=core_data.location_address or 'Not specified',
        coordinates=f'({core_data.coordinates[0]:.5f}, {core_data.coordinates[1]:.5f})' if core_data.coordinates else 'Not specified',
        zip_code=core_data.zip_code or 'Not specified',
        description=core_data.description or 'No specific description in this update.'
    )
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

    prompt = PROMPTS['FILL_EIDO_TEMPLATE'].format(
        retrieved_context_str=retrieved_context_str,
        scenario_desc=scenario_desc,
        template_str=template_str
    )
    logger.info("Calling LLM (RAG-augmented) to fill EIDO template.")
    response_text = _call_llm(prompt, is_json_output=True)

    if not response_text:
        logger.error("LLM call for template filling returned no response.")
        return None

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

    try:
        json.loads(response_text) 
        logger.info("LLM successfully filled EIDO template and response is valid JSON.")
        return response_text
    except json.JSONDecodeError as e:
        logger.error(f"LLM response for template filling was not valid JSON: {e}. Response: {response_text[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error validating template filler response: {e}", exc_info=True)
        return None


def extract_eido_from_alert_text(alert_text: str) -> Optional[str]:
    retrieved_context_str = ""
    if eido_retriever.is_ready:
        rag_query = f"EIDO schema fields relevant for parsing this emergency alert: {alert_text[:150]}"
        retrieved_chunks = eido_retriever.retrieve_context(rag_query, top_k=3) 
        if retrieved_chunks:
            retrieved_context_str = "\n\n**Relevant EIDO Schema Context (Use this to guide field extraction and naming):**\n---\n" + "\n---\n".join(retrieved_chunks) + "\n---"
            logger.debug("Augmenting alert parsing prompt with retrieved schema context.")
        else: logger.debug("No relevant schema context retrieved for alert parsing prompt.")
    else: logger.warning("EIDO Retriever not ready, cannot augment alert parsing prompt.")

    campus_context_hint = ""
    if "ucsd" in alert_text.lower() or "uc san diego" in alert_text.lower() or "geisel" in alert_text.lower() or "revelle" in alert_text.lower() or "warren mall" in alert_text.lower():
        campus_context_hint = "\nThis alert may pertain to the UC San Diego (UCSD) campus. Consider common UCSD locations like 'Geisel Library', 'Price Center', names of colleges ('Revelle College', 'Muir College', etc.), or specific building names if mentioned. If a named location is given (e.g., 'Warren Mall'), attempt to determine its coordinates if not explicitly provided."
        logger.debug("Added UCSD campus context hint to LLM prompt for alert parsing.")

    prompt = PROMPTS['EXTRACT_EIDO_FROM_ALERT_TEXT'].format(
        retrieved_context_str=retrieved_context_str,
        campus_context_hint=campus_context_hint,
        example_date=datetime.datetime.now().strftime('%Y-%m-%d'),
        alert_text=alert_text
    )
    logger.info("Calling LLM (RAG-augmented) to extract structured data from single alert text event.")
    response_text = _call_llm(prompt, is_json_output=True)

    if not response_text:
        logger.error("LLM call for alert parsing returned no response.")
        return None

    # This is a good place for a raw print for debugging purposes.
    print(f"\n--- LLM RAW RESPONSE FOR ALERT PARSING ---\n{response_text}\n-----------------------------------------\n")

    response_text = response_text.strip()
    if response_text.startswith("```json"): response_text = response_text[7:]
    if response_text.endswith("```"): response_text = response_text[:-3]
    response_text = response_text.strip()

    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        json_candidate = response_text[start_brace : end_brace + 1]
        try:
            # Validate it's a JSON object before returning
            parsed_candidate = json.loads(json_candidate)
            if isinstance(parsed_candidate, dict):
                return json_candidate
            else:
                logger.warning(f"LLM returned valid JSON but it was not a dictionary. Type: {type(parsed_candidate)}")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse cleaned JSON candidate from alert parser: {e}. Candidate string: {json_candidate[:500]}")
            return None
    else:
        logger.warning(f"Could not isolate JSON object in LLM response for alert parser. Raw response: {response_text[:200]}...")
        return None


def split_raw_text_into_events(raw_text: str) -> Optional[List[str]]:
    if not raw_text: return None

    prompt = PROMPTS['SPLIT_RAW_TEXT_INTO_EVENTS'].format(raw_text=raw_text)
    logger.info("Calling LLM to split raw text into potential events.")
    response_text = _call_llm(prompt, is_json_output=True) 

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
        return None 

    try:
        parsed_list = json.loads(response_text)
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            cleaned_list = [item for item in parsed_list if item.strip()]
            if not cleaned_list:
                 logger.warning("LLM splitter returned a list with only empty strings. Treating as single event.")
                 return None 
            if len(cleaned_list) > 1: logger.info(f"LLM successfully split text into {len(cleaned_list)} non-empty potential events.")
            else: logger.info("LLM indicated only one event in the text block (returned list with 1 non-empty item).")
            return cleaned_list
        else:
            logger.error(f"LLM response for splitting was not a valid JSON list of strings. Type: {type(parsed_list)}. Content: {str(parsed_list)[:200]}")
            return None 
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from LLM during event splitting: {e}. Response: {response_text[:500]}")
        return None 
    except Exception as e:
        logger.error(f"Unexpected error parsing splitter response: {e}", exc_info=True)
        return None 

def geocode_address_with_llm(address_text: str) -> Optional[Tuple[float, float]]:
    if not address_text or not isinstance(address_text, str):
        logger.warning("LLM geocoding: Invalid address text provided.")
        return None

    config = _get_current_llm_config()
    if config.get('llm_provider') == 'none':
        logger.info("LLM geocoding skipped: LLM provider is 'none'.")
        return None
    
    prompt = PROMPTS['GEOCODE_ADDRESS_WITH_LLM'].format(address_text=address_text)
    logger.info(f"Calling LLM for direct geocoding attempt: '{address_text[:100]}...'")
    response_text = _call_llm(prompt, is_json_output=True)

    if not response_text:
        logger.error(f"LLM geocoding failed: No response received for address '{address_text[:100]}...'")
        return None

    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    try:
        data = json.loads(response_text)
        if not isinstance(data, dict):
            logger.warning(f"LLM geocoding: Response was not a JSON object for '{address_text[:100]}...'. Response: {response_text[:200]}")
            return None

        lat_val = data.get("latitude")
        lon_val = data.get("longitude")

        if isinstance(lat_val, (float, int)) and isinstance(lon_val, (float, int)):
            lat, lon = float(lat_val), float(lon_val)
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                logger.info(f"LLM direct geocoding successful for '{address_text[:100]}...': ({lat}, {lon})")
                return (lat, lon)
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


def extract_geolocatable_clues(narrative_text: str) -> Optional[Dict[str, Any]]:
    if not narrative_text or not isinstance(narrative_text, str):
        logger.warning("LLM GeoClue Extraction: Invalid narrative text provided.")
        return None

    config = _get_current_llm_config()
    if config.get('llm_provider') == 'none':
        logger.info("LLM GeoClue Extraction skipped: LLM provider is 'none'.")
        return None

    geographic_context_prompt = "The incident likely occurred on or very near the UC San Diego (UCSD) campus in La Jolla, California, or the broader San Diego area. UCSD landmarks include Geisel Library, Price Center, various colleges (Revelle, Muir, Marshall, Warren, ERC, Sixth, Seventh, Eighth), RIMAC, specific building names or numbers, and features like 'Library Walk' or 'Sun God Lawn'."

    prompt = PROMPTS['EXTRACT_GEOLOCATABLE_CLUES'].format(
        geographic_context_prompt=geographic_context_prompt,
        narrative_text=narrative_text
    )
    logger.info(f"Calling LLM for geolocatable clue extraction from: '{narrative_text[:150]}...'")
    response_text = _call_llm(prompt, is_json_output=True)

    if not response_text:
        logger.error(f"LLM GeoClue Extraction: No response received for text '{narrative_text[:100]}...'")
        return None

    response_text = response_text.strip()
    if response_text.startswith("```json"): response_text = response_text[7:]
    if response_text.endswith("```"): response_text = response_text[:-3]
    response_text = response_text.strip()
    
    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        json_candidate = response_text[start_brace : end_brace + 1]
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                list_keys = ["explicit_addresses", "named_entities", "environmental_descriptors", "spatial_relationships", "potential_ambiguities"]
                for key in list_keys:
                    if key not in data:
                        data[key] = []
                
                if "overall_confidence_of_clues" not in data:
                    data["overall_confidence_of_clues"] = "Medium" 

                logger.info(f"LLM GeoClue Extraction successful for text '{narrative_text[:100]}...'. Found {len(data.get('named_entities',[]))} entities, {len(data.get('spatial_relationships',[]))} relations.")
                return data
            else:
                logger.warning(f"LLM GeoClue Extraction: Response was not a JSON dictionary. Type: {type(data)}. Response: {json_candidate[:200]}")
                return None
        except json.JSONDecodeError:
            logger.error(f"LLM GeoClue Extraction: Failed to parse JSON response. Response: {json_candidate[:200]}")
            return None
    else:
        logger.error(f"LLM GeoClue Extraction: Could not isolate JSON object in response. Raw response: {response_text[:200]}...")
        return None