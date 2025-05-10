# ui/app.py
import streamlit as st
import json
import os
import pandas as pd
import time
from datetime import datetime, timezone
import sys
import uuid
import logging
from io import StringIO
import logging.handlers
import pydeck as pdk
from streamlit_ace import st_ace
from typing import List, Dict, Optional, Union, Any 

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="EIDO Sentinel | AI Incident Processor Demo", # Updated title
    page_icon="🚨",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'http://localhost:8000', # Link to the new landing page
        'Report a bug': "https://github.com/LXString/eido-sentinel/issues",
        'About': "# EIDO Sentinel v0.8.1\nAI-Powered Emergency Incident Processor POC. Visit our main showcase page at http://localhost:8000"
    }
)

# --- Setup Python Path & Imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

modules_imported_successfully = True; import_error_message = ""; original_error = None
try:
    from config.settings import settings
    log_level_to_set = settings.log_level.upper() if hasattr(settings, 'log_level') else 'INFO'
    logging.basicConfig(
        level=log_level_to_set,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True 
    )
    logger_ui = logging.getLogger("EidoSentinelUIDemo") # Specific logger for UI
    logger_ui.info(f"UI Demo Logger initialized with level: {log_level_to_set}")

    from agent.agent_core import eido_agent_instance
    from services.storage import incident_store
    from data_models.schemas import Incident, ReportCoreData 
    from agent.llm_interface import fill_eido_template
except Exception as e:
    modules_imported_successfully = False; import_error_message = f"Setup Error: {e}"; original_error = e
    print(f"CRITICAL UI SETUP ERROR: {import_error_message}")
    if original_error: print(original_error)


if not modules_imported_successfully:
    st.error(f"🚨 **CRITICAL ERROR:** Failed during application setup.")
    st.error(f"**Details:** {import_error_message}")
    if original_error: st.exception(original_error)
    st.warning("Please ensure dependencies are installed (`pip install -r requirements.txt`), the RAG index is built (`python utils/rag_indexer.py`), and environment variables (`.env`) are correctly configured.")
    st.info("Run the app from the project's root directory: `streamlit run ui/app.py` or `./run_streamlit.sh`")
    st.stop()

# --- Log Capture Setup (for UI display) ---
log_stream = StringIO()
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S') 
stream_handler_ui = logging.StreamHandler(log_stream)
stream_handler_ui.setFormatter(log_formatter)
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == log_stream for h in root_logger.handlers):
    root_logger.addHandler(stream_handler_ui)
    logger_ui.debug("UI Log Capture StreamHandler added to root logger.")


# --- Global Variables & Constants ---
UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(UI_DIR, '..'))
SAMPLE_DIR = os.path.join(PROJECT_ROOT_DIR, 'sample_eido')
TEMPLATE_DIR = os.path.join(PROJECT_ROOT_DIR, 'eido_templates')
LOGO_PATH = os.path.join(PROJECT_ROOT_DIR, 'static', 'images', 'logo_icon_dark.png') # Use a logo from static
CUSTOM_CSS_PATH = os.path.join(UI_DIR, 'custom_styles.css')

# Determine landing page URL (FastAPI server)
LANDING_PAGE_URL = f"http://{settings.api_host if settings.api_host not in ['0.0.0.0', None] else 'localhost'}:{settings.api_port if settings.api_port else 8000}"


for dir_path in [SAMPLE_DIR, TEMPLATE_DIR]:
    if not os.path.exists(dir_path):
        try: os.makedirs(dir_path, exist_ok=True); logger_ui.info(f"Created directory: {dir_path}")
        except Exception as e: st.error(f"Failed to create directory {dir_path}: {e}"); logger_ui.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)


# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'log_messages': [], 'map_data': pd.DataFrame(columns=['lat', 'lon', 'incident_id', 'type', 'status']),
        'total_incidents': 0, 'active_incidents': 0, 'settings_saved': False,
        'total_reports_geo_checked': 0, 'reports_with_geo': 0,
        'clear_inputs_on_rerun': False, 'generated_eido_json': None,
        'filtered_incidents_cache': [], 
        'active_filters': {}, 
        'last_processed_alert_results': [], 
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    if 'active_filters' not in st.session_state or not isinstance(st.session_state.active_filters, dict):
         st.session_state.active_filters = {}

    settings_keys_to_sync_from_config = [
        'llm_provider', 'google_api_key', 'google_model_name',
        'openrouter_api_key', 'openrouter_model_name', 'openrouter_api_base_url',
        'local_llm_api_key', 'local_llm_model_name', 'local_llm_api_base_url',
        'geocoding_user_agent',
    ]
    for key in settings_keys_to_sync_from_config:
        if key not in st.session_state:
            st.session_state[key] = getattr(settings, key, None)

    if 'google_model_options' not in st.session_state:
        st.session_state.google_model_options = getattr(settings, 'google_model_options', ["gemini-1.5-flash-latest"])
        current_gmn = st.session_state.get('google_model_name', settings.google_model_name)
        if current_gmn not in st.session_state.google_model_options:
            logger_ui.warning(f"Session state google_model_name '{current_gmn}' not in options. Adding or defaulting.")
            if current_gmn and isinstance(st.session_state.google_model_options, list):
                 st.session_state.google_model_options.append(current_gmn) # Add if missing
                 st.session_state.google_model_options = sorted(list(set(st.session_state.google_model_options))) # Keep unique & sorted
            # Default to first available option or settings default
            st.session_state.google_model_name = st.session_state.google_model_options[0] if st.session_state.google_model_options else settings.google_model_name

init_session_state()

# --- Helper Functions ---
def list_files_in_dir(dir_path, extension=".json"):
    if not os.path.exists(dir_path): logger_ui.warning(f"Directory not found: {dir_path}"); return []
    try: return sorted([f for f in os.listdir(dir_path) if f.endswith(extension) and not f.startswith('.')])
    except Exception as e: st.error(f"Error listing files in {dir_path}: {e}"); logger_ui.error(f"Error listing files in {dir_path}: {e}", exc_info=True); return []

def get_captured_logs():
     log_stream.seek(0); logs_captured_this_run = log_stream.read(); log_stream.truncate(0); log_stream.seek(0)
     new_entries = [entry for entry in logs_captured_this_run.strip().split('\n') if entry.strip()]
     if new_entries:
         st.session_state.log_messages = new_entries + st.session_state.log_messages
         max_log_display = 200
         if len(st.session_state.log_messages) > max_log_display:
             st.session_state.log_messages = st.session_state.log_messages[:max_log_display]

def update_dashboard_metrics_and_cache():
    all_incidents_from_store = incident_store.get_all_incidents()
    st.session_state.total_incidents = len(all_incidents_from_store)

    active_statuses = ["active", "updated", "received", "rcvd", "dispatched", "dsp", "acknowledged", "ack", "enroute", "enr", "onscene", "onscn", "monitoring"]
    st.session_state.active_incidents = sum(1 for inc in all_incidents_from_store if inc.status and inc.status.lower() in active_statuses)

    total_geo_checked = 0; reports_with_coords = 0
    for inc in all_incidents_from_store:
        if inc.reports_core_data:
            for report_core in inc.reports_core_data:
                total_geo_checked +=1
                if report_core.coordinates and isinstance(report_core.coordinates, tuple) and len(report_core.coordinates) == 2:
                    try:
                        lat, lon = float(report_core.coordinates[0]), float(report_core.coordinates[1])
                        if -90 <= lat <= 90 and -180 <= lon <= 180: reports_with_coords +=1
                    except (ValueError, TypeError): pass
    st.session_state.total_reports_geo_checked = total_geo_checked
    st.session_state.reports_with_geo = reports_with_coords

    filtered_inc_list = all_incidents_from_store
    active_filters = st.session_state.get('active_filters', {}) 
    if active_filters.get('types'):
        filtered_inc_list = [inc for inc in filtered_inc_list if inc.incident_type in active_filters['types']]
    if active_filters.get('statuses'):
        filtered_inc_list = [inc for inc in filtered_inc_list if inc.status in active_filters['statuses']]
    if active_filters.get('zips'):
        filtered_inc_list = [inc for inc in filtered_inc_list if any(zip_code in active_filters['zips'] for zip_code in inc.zip_codes)]

    st.session_state.filtered_incidents_cache = sorted(
        filtered_inc_list,
        key=lambda x: x.last_updated_at if x.last_updated_at else datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )

def load_custom_css():
    if os.path.exists(CUSTOM_CSS_PATH):
        try:
            with open(CUSTOM_CSS_PATH, 'r') as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            logger_ui.debug("Custom CSS loaded.")
        except Exception as e:
            logger_ui.error(f"Error loading custom CSS from {CUSTOM_CSS_PATH}: {e}")
    else:
        logger_ui.warning(f"Custom CSS file not found at {CUSTOM_CSS_PATH}. Using default styles.")

load_custom_css()

# --- UI Rendering ---
if os.path.exists(LOGO_PATH): st.sidebar.image(LOGO_PATH, width=80) # Use a logo from static
else: st.sidebar.markdown("## 🚨 EIDO Sentinel")

st.sidebar.markdown(f"**[🏠 Main Showcase & Docs]({LANDING_PAGE_URL})**", unsafe_allow_html=True)
st.sidebar.caption("Interactive Demo Application")
st.sidebar.divider()
st.sidebar.header("Agent Controls")


with st.sidebar.expander("⚙️ Configure Agent", expanded=False):
    st.subheader("LLM Configuration"); st.caption("Overrides .env for this session.")
    llm_provider_options = ['google', 'openrouter', 'local', 'none']
    current_llm_provider = st.session_state.get('llm_provider', settings.llm_provider)
    if current_llm_provider not in llm_provider_options: current_llm_provider = settings.llm_provider
    llm_provider_idx = llm_provider_options.index(current_llm_provider)
    st.selectbox("☁️ LLM Provider:", options=llm_provider_options, index=llm_provider_idx, key='llm_provider')

    if st.session_state.llm_provider == 'google':
        st.text_input("🔑 Google API Key:", value=st.session_state.get('google_api_key', ""), type="password", key='google_api_key')
        g_model_opts = st.session_state.get('google_model_options', [settings.google_model_name])
        curr_g_model = st.session_state.get('google_model_name', settings.google_model_name)
        try:
            g_model_idx = g_model_opts.index(curr_g_model) if curr_g_model in g_model_opts else 0
        except ValueError: 
            g_model_idx = 0
            if g_model_opts: st.session_state.google_model_name = g_model_opts[0] 
            else: st.session_state.google_model_name = settings.google_model_name 
            logger_ui.warning(f"Corrected Google model index for UI. Model: {st.session_state.google_model_name}")
        st.selectbox("🧠 Google Model:", options=g_model_opts, index=g_model_idx, key='google_model_name')
    elif st.session_state.llm_provider == 'openrouter':
        st.text_input("🔑 OpenRouter API Key:", value=st.session_state.get('openrouter_api_key', ""), type="password", key='openrouter_api_key')
        st.text_input("🧠 OpenRouter Model:", value=st.session_state.get('openrouter_model_name', settings.openrouter_model_name), key='openrouter_model_name')
        st.text_input("🔗 OpenRouter Base URL:", value=st.session_state.get('openrouter_api_base_url', settings.openrouter_api_base_url), key='openrouter_api_base_url')
    elif st.session_state.llm_provider == 'local':
        st.text_input("🔗 Local LLM Base URL (e.g., http://localhost:11434/v1):", value=st.session_state.get('local_llm_api_base_url', settings.local_llm_api_base_url), key='local_llm_api_base_url')
        st.text_input("🧠 Local LLM Model (e.g., llama3:latest):", value=st.session_state.get('local_llm_model_name', settings.local_llm_model_name), key='local_llm_model_name')
        st.text_input("🔑 Local LLM API Key (optional):", value=st.session_state.get('local_llm_api_key', ""), type="password", key='local_llm_api_key')
    elif st.session_state.llm_provider == 'none': st.info("LLM features disabled.", icon="🚫")

    st.text_input("📍 Geocoding User Agent:", value=st.session_state.get('geocoding_user_agent', settings.geocoding_user_agent), key='geocoding_user_agent_display', disabled=True, help="Set in .env file. Critical for Nominatim.")

st.sidebar.divider(); st.sidebar.header("📥 Data Ingestion")
json_default_val = "" if st.session_state.clear_inputs_on_rerun else st.session_state.get('json_input_area_val', "")
alert_default_val = "" if st.session_state.clear_inputs_on_rerun else st.session_state.get('alert_text_input_area_val', "")
if st.session_state.clear_inputs_on_rerun: st.session_state.clear_inputs_on_rerun = False

ingest_tab1, ingest_tab2 = st.sidebar.tabs(["📄 EIDO JSON", "✍️ Raw Text"])
with ingest_tab1:
    uploaded_files = st.file_uploader("📁 Upload EIDO JSON File(s)", type="json", accept_multiple_files=True, key="file_uploader_key")
    json_input_area = st.text_area("📋 Paste EIDO JSON", value=json_default_val, height=150, key="json_input_area_val", placeholder='Paste EIDO Message JSON here (single object or list of objects)...')
    available_samples = list_files_in_dir(SAMPLE_DIR)
    selected_sample = st.selectbox("📜 Or Load Sample EIDO:", options=["-- Select Sample --"] + available_samples, key="sample_select_key", index=0)
with ingest_tab2:
    alert_text_input_area = st.text_area("💬 Paste Raw Alert Text", value=alert_default_val, height=200, key="alert_text_input_area_val", placeholder='ALERT: Vehicle collision at Main/Elm...\nUpdate: Road blocked...\nNew Call: Fire alarm...')

if st.sidebar.button("🚀 Process Inputs", type="primary", use_container_width=True):
    llm_needed_for_text = bool(alert_text_input_area.strip())
    provider = st.session_state.get('llm_provider', 'none')
    key_missing_error = None
    if provider == 'google' and not st.session_state.get('google_api_key'): key_missing_error = "Google API Key"
    if provider == 'openrouter' and not st.session_state.get('openrouter_api_key'): key_missing_error = "OpenRouter API Key"
    if provider == 'local' and (not st.session_state.get('local_llm_api_base_url') or not st.session_state.get('local_llm_model_name')): key_missing_error = "Local LLM URL/Model"

    if llm_needed_for_text and (provider == 'none' or key_missing_error):
        st.error(f"⚠️ LLM processing for raw text required but not configured. Missing: {key_missing_error or 'LLM Provider is None'}. Please check Agent Settings.", icon="⚙️")
    else:
        status_placeholder = st.sidebar.empty()
        status_placeholder.info("Processing inputs...")

        with st.spinner('Agent is processing inputs... This may take a moment.'):
            reports_from_json_sources = []; source_names = []
            pasted_json_val = st.session_state.json_input_area_val
            if pasted_json_val:
                try: reports_from_json_sources.append(json.loads(pasted_json_val)); source_names.append("Pasted JSON")
                except Exception as e: st.error(f"Error parsing Pasted JSON: {e}")

            selected_sample_val = st.session_state.sample_select_key
            if selected_sample_val != "-- Select Sample --":
                try:
                    with open(os.path.join(SAMPLE_DIR, selected_sample_val), 'r', encoding='utf-8') as f: reports_from_json_sources.append(json.load(f)); source_names.append(f"Sample: {selected_sample_val}")
                except Exception as e: st.error(f"Error loading sample '{selected_sample_val}': {e}")

            uploaded_files_val = st.session_state.get("file_uploader_key", []) 
            if uploaded_files_val:
                for uf in uploaded_files_val:
                    try: reports_from_json_sources.append(json.loads(uf.getvalue().decode("utf-8"))); source_names.append(f"File: {uf.name}")
                    except Exception as e: st.error(f"Error parsing file '{uf.name}': {e}")

            alert_text_to_process_str = st.session_state.alert_text_input_area_val.strip()

            if not reports_from_json_sources and not alert_text_to_process_str:
                status_placeholder.warning("No data provided for processing.")
            else:
                total_processed_ok = 0; total_errors = 0; processing_details = []
                final_json_reports_list = []
                for idx, loaded_data in enumerate(reports_from_json_sources):
                    s_name = source_names[idx] if idx < len(source_names) else "Unknown JSON"
                    if isinstance(loaded_data, list): final_json_reports_list.extend(item for item in loaded_data if isinstance(item, dict))
                    elif isinstance(loaded_data, dict): final_json_reports_list.append(loaded_data)
                    else: processing_details.append(f"⚠️ Skipped invalid data type from '{s_name}'.")

                for i, report_dict in enumerate(final_json_reports_list):
                    msg_id = report_dict.get('eidoMessageIdentifier', report_dict.get('$id', f"json_doc_{i+1}"))[:20]
                    try:
                        result = eido_agent_instance.process_report_json(report_dict)
                        if result.get('status', '').lower() == 'success': total_processed_ok += 1; processing_details.append(f"✅ EIDO JSON ({msg_id}...): Processed. Incident: {result.get('incident_id','')[:8]}")
                        else: total_errors += 1; processing_details.append(f"❌ EIDO JSON ({msg_id}...): {result.get('status', 'Failed')}")
                    except Exception as e: total_errors += 1; processing_details.append(f"💥 EIDO JSON ({msg_id}...): CRITICAL ERROR - {e}"); logger_ui.critical(f"EIDO JSON processing critical error: {e}", exc_info=True)

                st.session_state.last_processed_alert_results = []
                if alert_text_to_process_str:
                    try:
                        alert_results: Union[List[Dict[str, Any]], Dict[str, Any]] = eido_agent_instance.process_alert_text(alert_text_to_process_str)
                        alert_results_list = [alert_results] if isinstance(alert_results, dict) else (alert_results if isinstance(alert_results, list) else [])
                        st.session_state.last_processed_alert_results = alert_results_list

                        if not alert_results_list: total_errors +=1; processing_details.append("⚠️ Alert Text: Agent returned no results.")
                        else:
                            for idx, res_dict in enumerate(alert_results_list):
                                if isinstance(res_dict, dict):
                                    event_label = f"Event {idx+1}/{len(alert_results_list)}"
                                    status_msg = res_dict.get('status', 'Unknown'); msg_id_alert = res_dict.get('message_id', f'alert_event_{idx+1}')[:20]
                                    if status_msg.lower() == 'success': total_processed_ok += 1; processing_details.append(f"✅ Alert Text ({event_label}, {msg_id_alert}...): Processed. Incident: {res_dict.get('incident_id','')[:8]}")
                                    else: total_errors += 1; processing_details.append(f"❌ Alert Text ({event_label}, {msg_id_alert}...): {status_msg}")
                                else: 
                                    total_errors += 1; processing_details.append(f"❌ Alert Text (Event {idx+1}/{len(alert_results_list)}): Invalid result format - {type(res_dict)}")
                    except Exception as e: total_errors += 1; processing_details.append(f"💥 Alert Text: CRITICAL ERROR - {e}"); logger_ui.critical(f"Alert text processing critical error: {e}", exc_info=True)

                if total_errors > 0: status_placeholder.warning(f"Processed: {total_processed_ok} OK, {total_errors} failed.", icon="⚠️")
                else: status_placeholder.success(f"Processed: {total_processed_ok} inputs successfully.", icon="✅")
                st.session_state.processing_details_to_show = processing_details
                update_dashboard_metrics_and_cache(); get_captured_logs()
                st.session_state.clear_inputs_on_rerun = True; time.sleep(0.1); st.rerun()

st.sidebar.divider()
with st.sidebar.expander("⚠️ Admin Actions", expanded=False):
    if st.button("🗑️ Clear All Incidents", key="clear_all_inc_btn", use_container_width=True):
        count = len(incident_store.get_all_incidents())
        if count > 0: incident_store.clear_store(); st.success(f"Cleared {count} incidents.")
        else: st.info("Incident store is already empty.")
        keys_to_reset = ['filtered_incidents_cache', 'active_filters', 'last_processed_alert_results']
        for key in keys_to_reset:
             if key in st.session_state:
                 if isinstance(st.session_state[key], list): st.session_state[key] = []
                 elif isinstance(st.session_state[key], dict): st.session_state[key] = {}
                 else: init_session_state() 
        update_dashboard_metrics_and_cache(); get_captured_logs(); time.sleep(0.1); st.rerun()
st.sidebar.divider()
with st.sidebar.expander("📄 Processing Log", expanded=False):
    get_captured_logs()
    log_container = st.container(height=300)
    with log_container:
        if st.session_state.log_messages: st.code("\n".join(st.session_state.log_messages), language='log')
        else: st.caption("Log is empty or no new messages.")

col_title, col_logo_main = st.columns([0.85, 0.15]) # Placeholder for main page logo if needed
with col_title: st.title("📊 EIDO Sentinel Demo Dashboard"); st.caption(f"Interactive Application (v0.8.1)")
st.divider()

if 'processing_details_to_show' in st.session_state and st.session_state.processing_details_to_show:
    with st.expander("Last Processing Run Details", expanded=True):
        for detail_msg in st.session_state.processing_details_to_show:
            if "✅" in detail_msg: st.info(detail_msg)
            elif "⚠️" in detail_msg or "❌" in detail_msg: st.warning(detail_msg)
            else: st.error(detail_msg) 
    del st.session_state.processing_details_to_show 

update_dashboard_metrics_and_cache()

metric_cols = st.columns(4)
metric_cols[0].metric("📈 Total Incidents", st.session_state.total_incidents)
metric_cols[1].metric("🔥 Active Incidents", st.session_state.active_incidents)
all_incs_metric = incident_store.get_all_incidents()
report_counts_metric = [len(inc.reports_core_data) for inc in all_incs_metric if inc.reports_core_data]
avg_reports_val_metric = sum(report_counts_metric) / len(report_counts_metric) if report_counts_metric else 0
metric_cols[2].metric("📄 Avg Reports/Incident", f"{avg_reports_val_metric:.1f}")
geo_perc = (st.session_state.reports_with_geo / st.session_state.total_reports_geo_checked * 100) if st.session_state.total_reports_geo_checked > 0 else 0
metric_cols[3].metric("📍 Reports w/ Coords", f"{st.session_state.reports_with_geo}/{st.session_state.total_reports_geo_checked}", f"{geo_perc:.0f}%")
st.divider()

st.subheader("🔎 Filter & Analyze Incidents")
filter_col1, filter_col2, filter_col3 = st.columns([0.4, 0.3, 0.3])
all_incidents_for_options = incident_store.get_all_incidents()
available_types = sorted(list(set(inc.incident_type for inc in all_incidents_for_options if inc.incident_type)))
available_statuses = sorted(list(set(inc.status for inc in all_incidents_for_options if inc.status)))
available_zips = sorted(list(set(zip_code for inc in all_incidents_for_options for zip_code in inc.zip_codes if zip_code)))

def update_filters():
     st.session_state.active_filters['types'] = st.session_state.filter_type_ms
     st.session_state.active_filters['statuses'] = st.session_state.filter_status_ms
     st.session_state.active_filters['zips'] = st.session_state.filter_zip_ms

with filter_col1: st.multiselect("Filter by Type:", options=available_types, default=st.session_state.active_filters.get('types',[]), key="filter_type_ms", on_change=update_filters)
with filter_col2: st.multiselect("Filter by Status:", options=available_statuses, default=st.session_state.active_filters.get('statuses',[]), key="filter_status_ms", on_change=update_filters)
with filter_col3: st.multiselect("Filter by ZIP Code:", options=available_zips, default=st.session_state.active_filters.get('zips',[]), key="filter_zip_ms", on_change=update_filters)
st.divider()

# Main content tabs - Showcase tab is removed from here
tab_list, tab_map, tab_charts, tab_details, tab_warning, tab_eido_explorer, tab_eido_generator = st.tabs([
    "🗓️ **List**", "🗺️ **Map**", "📈 **Charts**", "🔍 **Details**", "📢 **Warnings**", "📄 **EIDO Explorer**", "📝 **EIDO Generator**"
])

filtered_incidents_to_display = st.session_state.filtered_incidents_cache

with tab_list: # Content remains the same
    st.caption(f"Displaying {len(filtered_incidents_to_display)} incidents based on filters.")
    if filtered_incidents_to_display:
        list_data = [{"ID": inc.incident_id[:8], "Type": inc.incident_type or "N/A", "Status": inc.status or "N/A",
                      "Last Update": pd.to_datetime(inc.last_updated_at, errors='coerce', utc=True),
                      "Reports": inc.trend_data.get('report_count', 0),
                      "Locations": len(inc.locations), "ZIPs": ", ".join(inc.zip_codes or []) or "N/A",
                      "Summary": inc.summary or "N/A"} for inc in filtered_incidents_to_display]
        df_list_display = pd.DataFrame(list_data)
        st.dataframe(df_list_display, use_container_width=True, hide_index=True,
                       column_order=("ID", "Type", "Status", "Last Update", "Reports", "Locations", "ZIPs", "Summary"),
                       column_config={
                           "ID": st.column_config.TextColumn("ID", width="small"),
                           "Type": st.column_config.TextColumn("Type", width="medium"),
                           "Status": st.column_config.TextColumn("Status", width="small"),
                           "Last Update": st.column_config.DatetimeColumn("Last Update", format="YYYY-MM-DD HH:mm Z", width="medium"),
                           "Reports": st.column_config.NumberColumn("Reports", format="%d", width="small"),
                           "Locations": st.column_config.NumberColumn("Locs", format="%d", width="small"),
                           "ZIPs": st.column_config.TextColumn("ZIPs", width="medium"),
                           "Summary": st.column_config.TextColumn("Summary", width="large")
                       }
        )
    else: st.info("No incidents match the current filter criteria.", icon="🚫")

with tab_map: # Content remains the same
    st.caption(f"Displaying locations for {len(filtered_incidents_to_display)} filtered incidents.")
    map_points_filtered = []
    for inc in filtered_incidents_to_display:
        if inc.locations:
            for lat, lon in inc.locations:
                 if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and -90 <= lat <= 90 and -180 <= lon <= 180:
                    map_points_filtered.append({'lat': lat, 'lon': lon, 'incident_id': inc.incident_id[:8], 'type': inc.incident_type or "Unknown", 'status': inc.status or "Unknown"})

    if map_points_filtered:
        map_df_filtered = pd.DataFrame(map_points_filtered)
        map_df_filtered.drop_duplicates(subset=['lat', 'lon', 'incident_id'], inplace=True) 

        try:
            mid_lat = map_df_filtered['lat'].median(); mid_lon = map_df_filtered['lon'].median()
            view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=10, pitch=45)
            scatter_layer = pdk.Layer('ScatterplotLayer', data=map_df_filtered, get_position='[lon, lat]', get_color='[200, 30, 0, 160]', get_radius=50, pickable=True, auto_highlight=True)
            heatmap_layer = pdk.Layer("HeatmapLayer", data=map_df_filtered, opacity=0.7, get_position=["lon", "lat"], aggregation=pdk.types.String("MEAN"), threshold=0.1, get_weight=1, pickable=False)
            tooltip = {"html": "<b>Incident:</b> {incident_id}<br/><b>Type:</b> {type}<br/><b>Status:</b> {status}", "style": {"backgroundColor": "steelblue", "color": "white"}}
            st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v10', initial_view_state=view_state, layers=[heatmap_layer, scatter_layer], tooltip=tooltip))
            with st.expander("Show Raw Map Data"): st.dataframe(map_df_filtered[['incident_id', 'type', 'status', 'lat', 'lon']].round(6), use_container_width=True, hide_index=True)
        except Exception as map_error:
             st.error(f"Error displaying PyDeck map: {map_error}")
    else: st.info("No geocoded locations match the current filter criteria.", icon="🗺️")

with tab_charts: # Content remains the same
    st.caption(f"Displaying charts for {len(filtered_incidents_to_display)} filtered incidents.")
    if filtered_incidents_to_display:
        chart_data = [{"Status": inc.status or "N/A", "Type": inc.incident_type or "N/A"} for inc in filtered_incidents_to_display]
        df_chart = pd.DataFrame(chart_data)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### Status Distribution")
            status_counts = df_chart['Status'].value_counts()
            if not status_counts.empty: st.bar_chart(status_counts)
            else: st.caption("No status data.")
        with chart_col2:
            st.markdown("##### Top Incident Types")
            type_counts = df_chart['Type'].value_counts().head(10)
            if not type_counts.empty: st.bar_chart(type_counts)
            else: st.caption("No type data.")
    else: st.info("No incidents match filters to display charts.", icon="📊")

with tab_details: # Content remains the same
    st.caption(f"Select one of the {len(filtered_incidents_to_display)} filtered incidents for details.")
    if filtered_incidents_to_display:
        details_map = {f"{inc.incident_id[:8]} - {inc.incident_type or 'N/A'}": inc.incident_id for inc in filtered_incidents_to_display}
        details_options = ["-- Select Incident --"] + list(details_map.keys())
        selected_label = st.selectbox("Select Filtered Incident:", options=details_options, index=0, key="detail_incident_selector")

        if selected_label != "-- Select Incident --":
            selected_full_id = details_map.get(selected_label)
            selected_incident: Optional[Incident] = incident_store.get_incident(selected_full_id) if selected_full_id else None

            if selected_incident:
                st.markdown(f"#### Incident `{selected_incident.incident_id[:8]}` Details"); st.divider()
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.markdown(f"**Type:** {selected_incident.incident_type or 'N/A'}")
                    st.markdown(f"**Status:** `{selected_incident.status or 'Unknown'}`")
                    st.markdown(f"**Reports:** {selected_incident.trend_data.get('report_count', 0)}")
                with info_col2:
                    created_dt = pd.to_datetime(selected_incident.created_at, utc=True, errors='coerce')
                    updated_dt = pd.to_datetime(selected_incident.last_updated_at, utc=True, errors='coerce')
                    st.markdown(f"**Created:** {created_dt.strftime('%Y-%m-%d %H:%M Z') if pd.notna(created_dt) else 'N/A'}")
                    st.markdown(f"**Updated:** {updated_dt.strftime('%Y-%m-%d %H:%M Z') if pd.notna(updated_dt) else 'N/A'}")
                    st.markdown(f"**ZIPs:** `{', '.join(selected_incident.zip_codes) or 'N/A'}`")

                st.divider(); st.markdown("##### 📄 AI Summary"); st.info(selected_incident.summary or "_No summary generated._", icon="🤖")
                st.markdown("##### ✅ Recommended Actions")
                if selected_incident.recommended_actions: st.markdown("\n".join(f"- {action}" for action in selected_incident.recommended_actions))
                else: st.caption("_No actions recommended._")
                st.divider()

                with st.expander(f"📜 Associated Reports ({len(selected_incident.reports_core_data or [])})", expanded=False):
                     if selected_incident.reports_core_data:
                         sorted_reports_core = sorted(
                             selected_incident.reports_core_data,
                             key=lambda r: r.timestamp if r.timestamp else datetime.min.replace(tzinfo=timezone.utc),
                             reverse=True 
                         )
                         report_data_list = []
                         for r_core in sorted_reports_core:
                             ts = pd.to_datetime(r_core.timestamp, errors='coerce', utc=True)
                             report_data_list.append({
                                 "Rept ID": r_core.report_id[:8], "Timestamp": ts,
                                 "Ext. ID": r_core.external_incident_id or "N/A", "Source": r_core.source or "N/A",
                                 "Description": r_core.description or "N/A", "Address": r_core.location_address or "N/A",
                                 "Coords": f"{r_core.coordinates[0]:.5f}, {r_core.coordinates[1]:.5f}" if r_core.coordinates else "N/A",
                                 "ZIP": r_core.zip_code or "N/A"
                             })
                         df_reports = pd.DataFrame(report_data_list)
                         st.dataframe(df_reports, hide_index=True, use_container_width=True,
                                       column_config={"Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss Z")})
                     else: st.info("No report data associated.")
                with st.expander(f"📋 Match/Update History", expanded=False):
                    match_info = selected_incident.trend_data.get('last_match_info', 'N/A')
                    st.caption(f"**Last Update Reason:** {match_info}")
            else: st.warning(f"Could not retrieve details for incident ID {selected_label}.")
    else: st.info("No incidents match filters to show details.", icon="🚫")

with tab_warning: # Content remains the same
    st.subheader("📢 Generate Incident Warning Text")
    st.caption(f"Based on the {len(filtered_incidents_to_display)} currently filtered incidents.")
    if filtered_incidents_to_display:
        warning_level = st.select_slider("Select Warning Severity:", options=["Informational", "Advisory", "Watch", "Warning"], value="Advisory", key="warn_level")
        custom_message = st.text_area("Add Custom Message (Optional):", height=100, key="warn_custom_msg")
        if st.button("📝 Generate Warning", use_container_width=True, key="warn_generate_btn"):
            warning_text = f"**--- {warning_level.upper()} ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M Z')}) ---**\n\n"
            if custom_message: warning_text += f"{custom_message}\n\n"
            warning_text += f"**Summary of Filtered Incidents ({len(filtered_incidents_to_display)}):**\n"
            for inc in filtered_incidents_to_display[:10]: # Limit to 10 for brevity
                warning_text += f"- **ID:** {inc.incident_id[:8]}, **Type:** {inc.incident_type or 'N/A'}, **Status:** {inc.status or 'N/A'}\n"
                loc_summary = ", ".join(inc.addresses[:1]) or (f"Coords: {inc.locations[0]}" if inc.locations else "N/A Location")
                warning_text += f"  Location: {loc_summary}\n"
                warning_text += f"  Summary: {inc.summary[:150]}...\n"
            if len(filtered_incidents_to_display) > 10: warning_text += f"\n... and {len(filtered_incidents_to_display) - 10} more incidents matching filters."
            st.text_area("Generated Warning Text (Copy below):", value=warning_text, height=300, key="warning_output_area")
    else: st.info("Apply filters to select incidents for warning generation.", icon="⚠️")

with tab_eido_explorer: # Content remains the same
    st.subheader("📄 Explore Original/Generated EIDO Data")
    st.caption("View the EIDO JSON associated with processed reports (original or LLM-generated).")
    if filtered_incidents_to_display:
        explorer_map = {f"{inc.incident_id[:8]} - {inc.incident_type or 'N/A'}": inc.incident_id for inc in filtered_incidents_to_display}
        explorer_options = ["-- Select Incident --"] + list(explorer_map.keys())
        selected_inc_label_exp = st.selectbox("Select Incident to Explore:", options=explorer_options, index=0, key="eido_explorer_inc_select")

        if selected_inc_label_exp != "-- Select Incident --":
            selected_inc_full_id_exp = explorer_map.get(selected_inc_label_exp)
            selected_inc_obj_exp: Optional[Incident] = incident_store.get_incident(selected_inc_full_id_exp) if selected_inc_full_id_exp else None

            if selected_inc_obj_exp and selected_inc_obj_exp.reports_core_data:
                sorted_reports_exp = sorted(selected_inc_obj_exp.reports_core_data, key=lambda r: r.timestamp if r.timestamp else datetime.min.replace(tzinfo=timezone.utc))
                report_options_exp = {}
                for idx, r_core in enumerate(sorted_reports_exp):
                    ts = pd.to_datetime(r_core.timestamp, errors='coerce', utc=True)
                    label = f"Report {idx+1} ({ts.strftime('%H:%M:%S Z') if pd.notna(ts) else 'No Time'}) - ID: {r_core.report_id[:8]}"
                    report_options_exp[label] = r_core.report_id 
                if not report_options_exp: st.warning("Selected incident has no associated reports to explore.")
                else:
                    selected_report_label_exp = st.selectbox(f"Select Report for Incident {selected_inc_obj_exp.incident_id[:8]}:", options=["-- Select Report --"] + list(report_options_exp.keys()), index=0, key=f"eido_explorer_report_select_{selected_inc_obj_exp.incident_id}")
                    if selected_report_label_exp != "-- Select Report --":
                        selected_report_id_exp = report_options_exp.get(selected_report_label_exp)
                        selected_report_core_exp = next((rc for rc in selected_inc_obj_exp.reports_core_data if rc.report_id == selected_report_id_exp), None)
                        if selected_report_core_exp and selected_report_core_exp.original_eido_dict:
                            eido_to_display = selected_report_core_exp.original_eido_dict
                            display_label = "Original/Generated EIDO JSON:"
                            try:
                                eido_str_display = json.dumps(eido_to_display, indent=2)
                                st.markdown(f"**{display_label}** (Report ID: `{selected_report_core_exp.report_id[:8]}`)")
                                st_ace(value=eido_str_display, language="json", theme="github", readonly=True, key=f"ace_editor_exp_{selected_report_id_exp}", height=400, wrap=True)
                                st.download_button(label="📥 Download this EIDO JSON", data=eido_str_display.encode('utf-8'), file_name=f"eido_report_{selected_report_core_exp.report_id[:8]}.json", mime="application/json", key=f"dl_eido_report_exp_{selected_report_id_exp}")
                            except Exception as json_err: st.error(f"Error formatting EIDO JSON for display: {json_err}"); st.json(eido_to_display) 
                        elif selected_report_core_exp: st.warning("Selected report does not have stored EIDO data.")
            elif selected_inc_obj_exp: st.info("Selected incident has no associated report data.")
    else: st.info("No incidents match filters to explore EIDO data.", icon="🚫")

with tab_eido_generator: # Content remains the same
    st.subheader("📝 Generate Compliant EIDO JSON"); st.info("Use LLM assistance to fill a standard EIDO template based on a scenario description.", icon="🤖")
    available_templates = list_files_in_dir(TEMPLATE_DIR)
    if not available_templates:
        st.warning(f"No EIDO templates found in `{TEMPLATE_DIR}`. Create JSON templates with placeholders like `[PLACEHOLDER]`.", icon="⚠️")
    else:
        template_options = ["-- Select Template --"] + available_templates
        selected_template_file = st.selectbox("Select EIDO Template:", options=template_options, index=0, key="generator_template_select")
        scenario_description = st.text_area("Enter Scenario Description:", height=150, key="generator_scenario_input", placeholder="Describe the incident (e.g., 'Structure fire at 100 Main St, Apt 5, reported by Engine 3 at 08:15 UTC...')")

        if st.button("✨ Generate EIDO from Template", key="generator_button", disabled=(selected_template_file == "-- Select Template --" or not scenario_description)):
            template_path = os.path.join(TEMPLATE_DIR, selected_template_file)
            try:
                with open(template_path, 'r', encoding='utf-8') as f: template_content = f.read()
                provider = st.session_state.get('llm_provider', 'none'); key_missing = False
                if provider == 'google' and not st.session_state.get('google_api_key'): key_missing = True
                if provider == 'openrouter' and not st.session_state.get('openrouter_api_key'): key_missing = True
                if provider == 'local' and (not st.session_state.get('local_llm_api_base_url') or not st.session_state.get('local_llm_model_name')): key_missing = True

                if provider == 'none' or key_missing: st.error("LLM is not configured. Please check Agent Settings.", icon="⚙️")
                else:
                    with st.spinner("Generating EIDO JSON..."):
                        generated_json_str = fill_eido_template(template_content, scenario_description)
                        if generated_json_str:
                            st.session_state.generated_eido_json = generated_json_str
                            st.success("EIDO JSON generated successfully!", icon="✅")
                        else:
                            st.session_state.generated_eido_json = None
                            st.error("Failed to generate EIDO JSON from template using LLM. Check logs.", icon="❌")
                        get_captured_logs()
            except FileNotFoundError: st.error(f"Template file not found: {selected_template_file}")
            except Exception as e: st.error(f"Error generating EIDO: {e}"); logger_ui.error(f"EIDO Generator error: {e}", exc_info=True); st.session_state.generated_eido_json = None

        if st.session_state.generated_eido_json:
            st.markdown("---"); st.markdown("**Generated EIDO JSON:**")
            try:
                parsed_generated_json = json.loads(st.session_state.generated_eido_json)
                pretty_json_str = json.dumps(parsed_generated_json, indent=2)
                st_ace(value=pretty_json_str, language="json", theme="github", keybinding="vscode", font_size=12, height=400, show_gutter=True, show_print_margin=False, wrap=True, auto_update=False, readonly=True, key="ace_editor_generator_output")
                st.download_button(label="📥 Download Generated EIDO", data=pretty_json_str.encode('utf-8'), file_name=f"generated_{selected_template_file.replace('.json','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", mime="application/json", key="dl_generated_eido")
            except Exception as display_err: st.error(f"Error displaying/formatting generated EIDO: {display_err}"); st.text_area("Raw Generated Output:", value=st.session_state.generated_eido_json, height=400)

st.divider()
st.caption(f"EIDO Sentinel v0.8.1 | Interactive Demo Application")
get_captured_logs()