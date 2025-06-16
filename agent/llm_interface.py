import logging
from typing import List, Optional, Dict, Any, Tuple
import sys
import os
import json
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
    raise SystemExit(
        f"llm_interface failed to import dependencies: {e}") from e

logger = logging.getLogger(__name__)


def load_prompts_from_json() -> Dict[str, str]:
    """Loads all prompts from the JSON library file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "prompt_library.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        logger.info(
            f"Successfully loaded {len(prompts)} prompts from {json_path}")
        return prompts
    except Exception as e:
        logger.critical(
            f"FATAL: An unexpected error occurred while loading prompts: {e}")
        raise


PROMPTS = load_prompts_from_json()
_client_cache: Dict[tuple, Any] = {}


def _get_llm_client(config: Dict[str, Any]) -> Optional[Any]:
    provider = config.get('llm_provider')
    cache_key = None
    client = None

    if not provider or provider == 'none':
        return None

    try:
        if provider == 'google':
            api_key = config.get('google_api_key')
            if not api_key:
                return None
            model_name = config.get('google_model_name')
            cache_key = (provider, api_key, model_name)
            if cache_key in _client_cache:
                return _client_cache[cache_key]

            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(model_name)
            _client_cache[cache_key] = client
            return client

        elif provider == 'openrouter':
            api_key = config.get('openrouter_api_key')
            base_url = config.get('openrouter_api_base_url')
            if not api_key or not base_url:
                return None
            cache_key = (provider, api_key, base_url)
            if cache_key in _client_cache:
                return _client_cache[cache_key]

            client = OpenAI(api_key=api_key, base_url=base_url)
            _client_cache[cache_key] = client
            return client

        elif provider == 'local':
            api_key = config.get('local_llm_api_key', 'EMPTY')
            base_url = config.get('local_llm_api_base_url')
            if not base_url:
                return None
            cache_key = (provider, api_key, base_url)
            if cache_key in _client_cache:
                return _client_cache[cache_key]

            client = OpenAI(api_key=api_key, base_url=base_url)
            _client_cache[cache_key] = client
            return client
        else:
            logger.error(f"Unsupported LLM provider in config: {provider}")
            return None
    except Exception as e:
        safe_config = {k: v for k,
                       v in config.items() if 'key' not in k.lower()}
        logger.critical(
            f"Failed to configure or get LLM client for provider '{provider}'. Config (sanitized): {json.dumps(safe_config, default=str)}. Error: {e}", exc_info=True)
        return None


def _get_current_llm_config() -> Dict[str, Any]:
    """
    Returns the application's current LLM configuration from the main settings object.
    This ensures the backend is self-contained and not dependent on UI state.
    """
    logger.debug("Using initial settings for LLM config.")
    return initial_settings.model_dump()


def _call_llm(prompt: str, is_json_output: bool = False) -> Optional[str]:
    logger.debug(f"--- Attempting LLM call --- JSON Mode: {is_json_output}")
    config = _get_current_llm_config()
    provider = config.get('llm_provider')
    llm_client = _get_llm_client(config)

    if not llm_client:
        logger.error(
            f"LLM client for provider '{provider}' could not be initialized. Cannot call LLM.")
        return None

    model_name_map = {
        'google': config.get('google_model_name'),
        'openrouter': config.get('openrouter_model_name'),
        'local': config.get('local_llm_model_name')
    }
    model_name = model_name_map.get(provider)

    if not model_name:
        logger.error(
            f"LLM model name is not configured for provider '{provider}'. Cannot call LLM.")
        return None

    logger.info(f"Calling LLM Provider: {provider}, Model: {model_name}")

    try:
        if provider == 'google':
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json") if is_json_output else None
            response = llm_client.generate_content(
                prompt, generation_config=generation_config)

            if response and hasattr(response, 'text') and response.text:
                logger.info("LLM call successful (Google).")
                return response.text.strip()
            else:
                failure_reason = getattr(
                    response, 'prompt_feedback', 'Response was empty or malformed.')
                logger.error(
                    f"LLM call failed (Google). Reason: {failure_reason}")
                return None

        elif provider in ['openrouter', 'local']:
            api_name = provider.capitalize()
            messages = [{"role": "system", "content": "You are an AI assistant. If JSON output is requested, provide ONLY the valid JSON object."}, {
                "role": "user", "content": prompt}]
            request_params = {"model": model_name, "messages": messages}
            if is_json_output:
                request_params["response_format"] = {"type": "json_object"}

            response = llm_client.chat.completions.create(**request_params)
            if response and response.choices and response.choices[0].message.content:
                logger.info(f"LLM call successful ({api_name}).")
                return response.choices[0].message.content.strip()
            else:
                logger.error(
                    f"LLM call failed or returned empty choices ({api_name}). Finish Reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")
                return None

    except Exception as e:
        api_error_details = getattr(e, 'message', str(e))
        logger.error(
            f"Error calling LLM API ({provider} - {model_name}): {type(e).__name__}: {api_error_details}", exc_info=True)
        return None
    return None


def _clean_json_response(response_text: str) -> Optional[str]:
    """Helper to extract a JSON object or list from a string."""
    if not response_text:
        return None

    response_text = response_text.strip()

    # Find the first '{' or '[' and the last '}' or ']'
    start_brace = response_text.find('{')
    start_bracket = response_text.find('[')

    first_char_pos = -1
    if start_brace != -1 and start_bracket != -1:
        first_char_pos = min(start_brace, start_bracket)
    elif start_brace != -1:
        first_char_pos = start_brace
    elif start_bracket != -1:
        first_char_pos = start_bracket

    if first_char_pos == -1:
        return None

    last_brace = response_text.rfind('}')
    last_bracket = response_text.rfind(']')
    last_char_pos = max(last_brace, last_bracket)

    if last_char_pos <= first_char_pos:
        return None

    json_candidate = response_text[first_char_pos: last_char_pos + 1]

    try:
        json.loads(json_candidate)
        return json_candidate
    except json.JSONDecodeError:
        logger.warning(
            f"Could not parse cleaned JSON candidate. Raw response prefix: {response_text[:100]}")
        return None


def summarize_incident(history: str, core_data: ReportCoreData) -> Optional[str]:
    prompt = PROMPTS['SUMMARIZE_INCIDENT'].format(
        history=history if history else "No previous history available.",
        timestamp=core_data.timestamp.isoformat(
            timespec='seconds').replace('+00:00', 'Z'),
        incident_type=core_data.incident_type or 'Not specified',
        source=core_data.source or 'Unknown',
        location_address=core_data.location_address or 'Not specified',
        coordinates=f'({core_data.coordinates[0]:.5f}, {core_data.coordinates[1]:.5f})' if core_data.coordinates else 'Not specified',
        zip_code=core_data.zip_code or 'Not specified',
        description=core_data.description or 'No specific description in this update.'
    )
    logger.debug(
        f"Calling LLM for incident summary (Report ID: {core_data.report_id[:8]})")
    return _call_llm(prompt)


def recommend_actions(summary: str, core_data: ReportCoreData) -> Optional[List[str]]:
    prompt = PROMPTS['RECOMMEND_ACTIONS'].format(
        summary=summary,
        timestamp=core_data.timestamp.isoformat(
            timespec='seconds').replace('+00:00', 'Z'),
        incident_type=core_data.incident_type or 'Not specified',
        source=core_data.source or 'Unknown',
        location_address=core_data.location_address or 'Not specified',
        coordinates=f'({core_data.coordinates[0]:.5f}, {core_data.coordinates[1]:.5f})' if core_data.coordinates else 'Not specified',
        zip_code=core_data.zip_code or 'Not specified',
        description=core_data.description or 'No specific description in this update.'
    )
    logger.debug(
        f"Calling LLM for recommended actions (Report ID: {core_data.report_id[:8]})")
    response_text = _call_llm(prompt)
    if response_text:
        actions = [line.strip('- ').strip() for line in response_text.splitlines()
                   if line.strip() and line.strip().startswith('-')]
        return actions if actions else [f"AI Response: {response_text}"]
    return None


def fill_eido_template(template_str: str, scenario_desc: str) -> Optional[str]:
    if not template_str or not scenario_desc:
        return None
    rag_query = f"EIDO schema definitions relevant to: {scenario_desc[:150]}"
    retrieved_chunks = eido_retriever.retrieve_context(rag_query, top_k=3)
    retrieved_context_str = "\n\n**Relevant EIDO Schema Context:**\n---\n" + \
        "\n---\n".join(retrieved_chunks) + "\n---" if retrieved_chunks else ""
    prompt = PROMPTS['FILL_EIDO_TEMPLATE'].format(
        retrieved_context_str=retrieved_context_str, scenario_desc=scenario_desc, template_str=template_str)

    response_text = _call_llm(prompt, is_json_output=True)
    return _clean_json_response(response_text)


def choose_eido_template(alert_text: str, template_summaries_str: str) -> Optional[str]:
    """Given an alert text and a list of template summaries, chooses the best template filename."""
    if not alert_text or not template_summaries_str:
        return None

    prompt = PROMPTS['CHOOSE_EIDO_TEMPLATE'].format(
        template_summaries_str=template_summaries_str,
        alert_text=alert_text
    )
    logger.debug(
        f"Calling LLM to choose an EIDO template for alert: '{alert_text[:50]}...'")

    # The output is a simple string, not JSON
    response_text = _call_llm(prompt, is_json_output=False)

    if response_text:
        # Clean up the response to get just the filename or NONE
        cleaned_response = response_text.strip().replace(
            '`', '').replace('"', '').replace("'", "")
        if cleaned_response.upper() == 'NONE':
            logger.info(
                "LLM determined no suitable EIDO template for the alert.")
            return None
        # Basic check to see if it looks like a filename
        if ".json" in cleaned_response:
            logger.info(f"LLM chose EIDO template: {cleaned_response}")
            return cleaned_response
        else:
            logger.warning(
                f"LLM returned an unexpected value for template choice: '{response_text}'")

    return None


def extract_eido_from_alert_text(alert_text: str) -> Optional[str]:
    rag_query = f"EIDO schema fields for parsing: {alert_text[:150]}"
    retrieved_chunks = eido_retriever.retrieve_context(rag_query, top_k=3)
    retrieved_context_str = "\n\n**Relevant EIDO Schema Context:**\n---\n" + \
        "\n---\n".join(retrieved_chunks) + "\n---" if retrieved_chunks else ""
    campus_context_hint = "\nThis alert may pertain to the UC San Diego (UCSD) campus. Consider common UCSD locations and landmarks." if any(
        keyword in alert_text.lower() for keyword in ["ucsd", "uc san diego", "geisel", "revelle"]) else ""
    prompt = PROMPTS['EXTRACT_EIDO_FROM_ALERT_TEXT'].format(
        retrieved_context_str=retrieved_context_str, campus_context_hint=campus_context_hint,
        example_date=datetime.datetime.now().strftime('%Y-%m-%d'), alert_text=alert_text
    )

    response_text = _call_llm(prompt, is_json_output=True)
    return _clean_json_response(response_text)


def split_raw_text_into_events(raw_text: str) -> Optional[List[str]]:
    if not raw_text:
        return None
    prompt = PROMPTS['SPLIT_RAW_TEXT_INTO_EVENTS'].format(raw_text=raw_text)
    response_text = _call_llm(prompt, is_json_output=True)
    cleaned_response = _clean_json_response(response_text)
    if cleaned_response:
        try:
            parsed_list = json.loads(cleaned_response)
            if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                cleaned_list = [item for item in parsed_list if item.strip()]
                return cleaned_list if cleaned_list else None
        except (json.JSONDecodeError, TypeError):
            logger.error(
                f"LLM response for splitting was not a valid JSON list of strings. Response: {cleaned_response[:200]}")
    return None


def geocode_address_with_llm(address_text: str) -> Optional[Tuple[float, float]]:
    if not address_text or not isinstance(address_text, str):
        return None
    config = _get_current_llm_config()
    if config.get('llm_provider') == 'none':
        return None

    prompt = PROMPTS['GEOCODE_ADDRESS_WITH_LLM'].format(
        address_text=address_text)
    response_text = _call_llm(prompt, is_json_output=True)
    cleaned_response = _clean_json_response(response_text)
    if cleaned_response:
        try:
            data = json.loads(cleaned_response)
            if isinstance(data, dict):
                lat, lon = data.get("latitude"), data.get("longitude")
                if isinstance(lat, (float, int)) and isinstance(lon, (float, int)) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return (lat, lon)
        except (json.JSONDecodeError, TypeError):
            logger.error(
                f"LLM response for geocoding was not valid JSON dictionary with lat/lon. Response: {cleaned_response[:200]}")
    return None


def extract_geolocatable_clues(narrative_text: str) -> Optional[Dict[str, Any]]:
    if not narrative_text or not isinstance(narrative_text, str):
        return None
    config = _get_current_llm_config()
    if config.get('llm_provider') == 'none':
        return None

    geographic_context_prompt = "The incident likely occurred on or very near the UC San Diego (UCSD) campus in La Jolla, California, or the broader San Diego area."
    prompt = PROMPTS['EXTRACT_GEOLOCATABLE_CLUES'].format(
        geographic_context_prompt=geographic_context_prompt, narrative_text=narrative_text)
    response_text = _call_llm(prompt, is_json_output=True)
    cleaned_response = _clean_json_response(response_text)
    if cleaned_response:
        try:
            data = json.loads(cleaned_response)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            logger.error(
                f"LLM response for geo-clue extraction was not a valid JSON dictionary. Response: {cleaned_response[:200]}")
    return None
