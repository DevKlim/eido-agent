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
from io import StringIO, BytesIO 
import logging.handlers
import pydeck as pdk
from streamlit_ace import st_ace
from typing import List, Dict, Optional, Union, Any 
import requests # New import for API calls

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="EIDO Sentinel | AI Incident Processor Demo",
    page_icon="img/logo_icon_light.png", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'http://localhost:8000', 
        'Report a bug': "https://github.com/LXString/eido-sentinel/issues",
        'About': "# EIDO Sentinel v0.9.1\nAI-Powered Emergency Incident Processor. Visit our main showcase at http://localhost:8000"
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
    logger_ui = logging.getLogger("EidoSentinelUIDemo") 
    logger_ui.info(f"UI Demo Logger initialized with level: {log_level_to_set}")

    # Agent and store are no longer directly imported for core logic
    # from agent.agent_core import eido_agent_instance 
    # from services.storage import incident_store
    from data_models.schemas import Incident as PydanticIncident, ReportCoreData # Still needed for type hints, display
    from utils.ocr_processor import extract_text_from_image 
except Exception as e:
    modules_imported_successfully = False; import_error_message = f"Setup Error: {e}"; original_error = e
    print(f"CRITICAL UI SETUP ERROR: {import_error_message}")
    if original_error: print(original_error)


if not modules_imported_successfully:
    st.error(f"CRITICAL ERROR: Failed during application setup.")
    st.error(f"Details: {import_error_message}")
    if original_error: st.exception(original_error)
    st.warning("Please ensure dependencies are installed (`pip install -r requirements.txt`), Tesseract OCR is installed, the RAG index is built (`python utils/rag_indexer.py`), and environment variables (`.env`) are correctly configured.")
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
LOGO_PATH = os.path.join(PROJECT_ROOT_DIR, 'static', 'images', 'logo_icon_dark.png') 
CUSTOM_CSS_PATH = os.path.join(UI_DIR, 'custom_styles.css')

LANDING_PAGE_URL = f"http://{settings.api_host if settings.api_host and settings.api_host != '0.0.0.0' else 'localhost'}:{settings.api_port if settings.api_port else 8000}"

API_BASE_URL = settings.api_base_url # Get from settings

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
        'ocr_text_output': "" ,
        'all_incidents_from_api': [] # Cache for API fetched incidents
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    if 'active_filters' not in st.session_state or not isinstance(st.session_state.active_filters, dict):
         st.session_state.active_filters = {}

    # LLM settings are managed by the backend, Streamlit UI doesn't need to send them for processing
    # It might need them if it makes direct LLM calls for UI-specific features NOT related to core agent processing
    # For this version, let's assume all core processing (including LLM use) is via FastAPI backend.
    # So, removing direct sync of LLM keys/models from session state here.
    # The agent configuration expander will be removed or modified.
    # For EIDO Generator, it will call a backend endpoint.
    
    # Agent config will be managed by backend's .env or platform env vars.
    # If UI needs to *display* current backend config, an API endpoint could provide that.
    # For simplicity, removing direct LLM config from UI for now.
    st.session_state.api_base_url = API_BASE_URL # Store API base URL in session state

init_session_state()

# --- API Helper Functions ---
def make_api_request(method: str, endpoint: str, payload: Optional[Dict] = None, params: Optional[Dict] = None) -> Optional[Any]:
    url = f"{st.session_state.api_base_url}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=payload, timeout=60) # Longer timeout for processing
        elif method.upper() == "PUT":
            response = requests.put(url, json=payload, timeout=30)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        if response.content: # Check if there is content to decode
            return response.json()
        return None # No content but successful (e.g., 204 No Content)
    except requests.exceptions.HTTPError as errh:
        st.error(f"API HTTP Error: {errh.response.status_code} - {errh.response.text}")
        logger_ui.error(f"API HTTP Error: {errh.response.status_code} for {url}. Response: {errh.response.text}", exc_info=True)
    except requests.exceptions.ConnectionError as errc:
        st.error(f"API Connection Error: {errc}. Is the backend server running at {st.session_state.api_base_url}?")
        logger_ui.error(f"API Connection Error for {url}: {errc}", exc_info=True)
    except requests.exceptions.Timeout as errt:
        st.error(f"API Timeout Error: {errt}. The request to {url} timed out.")
        logger_ui.error(f"API Timeout Error for {url}: {errt}", exc_info=True)
    except requests.exceptions.RequestException as err:
        st.error(f"API Request Error: {err}")
        logger_ui.error(f"API Request Error for {url}: {err}", exc_info=True)
    except json.JSONDecodeError as j_err:
        st.error(f"API JSON Decode Error: {j_err}. Response was not valid JSON.")
        logger_ui.error(f"API JSON Decode Error for {url}. Response: {response.text[:200]}...", exc_info=True)
    return None

# --- Helper Functions --- (Adapted for API calls)
def list_files_in_dir(dir_path, extension=".json"): # This remains local to UI for loading samples
    if not os.path.exists(dir_path): logger_ui.warning(f"Directory not found: {dir_path}"); return []
    try: return sorted([f for f in os.listdir(dir_path) if f.endswith(extension) and not f.startswith('.')])
    except Exception as e: st.error(f"Error listing files in {dir_path}: {e}"); logger_ui.error(f"Error listing files in {dir_path}: {e}", exc_info=True); return []

def get_captured_logs(): # Remains same
     log_stream.seek(0); logs_captured_this_run = log_stream.read(); log_stream.truncate(0); log_stream.seek(0)
     new_entries = [entry for entry in logs_captured_this_run.strip().split('\n') if entry.strip()]
     if new_entries:
         st.session_state.log_messages = new_entries + st.session_state.log_messages
         max_log_display = 200
         if len(st.session_state.log_messages) > max_log_display:
             st.session_state.log_messages = st.session_state.log_messages[:max_log_display]

def fetch_all_incidents_from_api():
    data = make_api_request("GET", "/api/v1/incidents")
    if data is not None:
        try:
            # Convert list of dicts to list of PydanticIncident objects
            st.session_state.all_incidents_from_api = [PydanticIncident(**inc) for inc in data]
        except Exception as e:
            st.error(f"Error parsing incidents from API: {e}")
            logger_ui.error(f"Error parsing incidents from API response: {e}. Data: {str(data)[:500]}", exc_info=True)
            st.session_state.all_incidents_from_api = []
    else: # API request failed
        st.session_state.all_incidents_from_api = [] # Ensure it's empty on failure

def update_dashboard_metrics_and_cache():
    # Fetch data from API if not already cached or if refresh is needed
    # For a demo, fetching on each update_dashboard_metrics call is okay.
    fetch_all_incidents_from_api()
    
    all_incidents = st.session_state.all_incidents_from_api
    st.session_state.total_incidents = len(all_incidents)

    active_statuses = ["active", "updated", "received", "rcvd", "dispatched", "dsp", "acknowledged", "ack", "enroute", "enr", "onscene", "onscn", "monitoring"]
    st.session_state.active_incidents = sum(1 for inc in all_incidents if inc.status and inc.status.lower() in active_statuses)

    total_geo_checked = 0; reports_with_coords = 0
    for inc in all_incidents:
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

    filtered_inc_list = all_incidents
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

def load_custom_css(): # Remains same
    if os.path.exists(CUSTOM_CSS_PATH):
        try:
            with open(CUSTOM_CSS_PATH, 'r') as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            logger_ui.debug("Custom CSS loaded.")
        except Exception as e:
            logger_ui.error(f"Error loading custom CSS from {CUSTOM_CSS_PATH}: {e}")
    else:
        logger_ui.warning(f"Custom CSS file not found at {CUSTOM_CSS_PATH}. Applying basic theme adjustments.")
        st.markdown("""
        <style>
            html, body, [class*="st-"], .stApp { font-family: 'Urbanist', sans-serif !important; }
             :root { /* Basic theming */ }
            [data-testid="stSidebar"] .stMarkdown p, 
            [data-testid="stSidebar"] label { color: #D0D9E4 !important; font-family: 'Urbanist', sans-serif !important; }
             [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
                font-family: 'Urbanist', sans-serif !important; color: #E0E7FF !important; }
        </style>
        """, unsafe_allow_html=True)

load_custom_css()

# --- UI Rendering ---
if os.path.exists(LOGO_PATH): st.sidebar.image(LOGO_PATH, width=60) 
else: st.sidebar.markdown("## EIDO Sentinel")

st.sidebar.markdown(f"**[Main Showcase Page]({LANDING_PAGE_URL})**", unsafe_allow_html=True)
st.sidebar.caption("AI Incident Processor - Interactive Demo")
st.sidebar.divider()
st.sidebar.header("Agent Status")
st.sidebar.info(f"Backend API: {st.session_state.api_base_url}")
# Simple check for API health (optional, can be slow)
# if st.sidebar.button("Check API Health"):
#     if make_api_request("GET", "/api/v1/incidents"): # Using /incidents as a basic health check endpoint
#         st.sidebar.success("API Connected")
#     else:
#         st.sidebar.error("API Connection Failed")

st.sidebar.divider(); st.sidebar.header("Data Ingestion")
json_default_val = "" if st.session_state.clear_inputs_on_rerun else st.session_state.get('json_input_area_val', "")
alert_default_val = "" if st.session_state.clear_inputs_on_rerun else st.session_state.get('alert_text_input_area_val', "")
if st.session_state.clear_inputs_on_rerun: st.session_state.clear_inputs_on_rerun = False

ingest_tab1, ingest_tab2, ingest_tab3 = st.sidebar.tabs(["EIDO JSON", "Raw Text", "Image (OCR)"]) 
with ingest_tab1:
    uploaded_files = st.file_uploader("Upload EIDO JSON File(s)", type="json", accept_multiple_files=True, key="file_uploader_key")
    json_input_area = st.text_area("Paste EIDO JSON", value=json_default_val, height=150, key="json_input_area_val", placeholder='Paste EIDO Message JSON here (single object or list of objects)...')
    available_samples = list_files_in_dir(SAMPLE_DIR)
    selected_sample = st.selectbox("Or Load Sample EIDO:", options=["-- Select Sample --"] + available_samples, key="sample_select_key", index=0)
with ingest_tab2:
    alert_text_input_area = st.text_area("Paste Raw Alert Text", value=alert_default_val, height=200, key="alert_text_input_area_val", placeholder='ALERT: Vehicle collision at Main/Elm...\nUpdate: Road blocked...\nNew Call: Fire alarm...')
with ingest_tab3:
    uploaded_image_ocr = st.file_uploader("Upload Image for OCR", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="ocr_image_uploader")
    if uploaded_image_ocr:
        if st.button("Extract Text from Image", key="ocr_extract_button"):
            with st.spinner("Performing OCR..."):
                image_bytes = BytesIO(uploaded_image_ocr.getvalue())
                ocr_text = extract_text_from_image(image_bytes) # This is a local util, still fine
                if ocr_text:
                    st.session_state.ocr_text_output = ocr_text
                    st.success("OCR successful! Text copied to 'Raw Alert Text' area below for processing.")
                    st.session_state.alert_text_input_area_val = ocr_text 
                    get_captured_logs(); time.sleep(0.1); st.rerun() 
                else:
                    st.error("OCR failed. Ensure Tesseract is installed and configured, or check image quality.")
                get_captured_logs()
    if st.session_state.get('ocr_text_output'):
        st.text_area("OCR Output (auto-copied to Raw Text):", value=st.session_state.ocr_text_output, height=100, disabled=True, key="ocr_output_display")


if st.sidebar.button("Process Inputs", type="primary", use_container_width=True):
    alert_text_to_process_str = st.session_state.alert_text_input_area_val.strip()
    if not alert_text_to_process_str and st.session_state.get('ocr_text_output', '').strip(): 
        alert_text_to_process_str = st.session_state.get('ocr_text_output', '').strip()
        logger_ui.info("Using OCR output as primary raw text for processing.")
    
    status_placeholder = st.sidebar.empty()
    status_placeholder.info("Sending inputs to backend for processing...")

    with st.spinner('Agent is processing inputs via API... This may take a moment.'):
        reports_from_json_sources = []; 
        pasted_json_val = st.session_state.json_input_area_val
        if pasted_json_val:
            try: reports_from_json_sources.append(json.loads(pasted_json_val))
            except Exception as e: st.error(f"Error parsing Pasted JSON: {e}")

        selected_sample_val = st.session_state.sample_select_key
        if selected_sample_val != "-- Select Sample --":
            try:
                with open(os.path.join(SAMPLE_DIR, selected_sample_val), 'r', encoding='utf-8') as f: reports_from_json_sources.append(json.load(f))
            except Exception as e: st.error(f"Error loading sample '{selected_sample_val}': {e}")

        uploaded_files_val = st.session_state.get("file_uploader_key", []) 
        if uploaded_files_val:
            for uf in uploaded_files_val:
                try: reports_from_json_sources.append(json.loads(uf.getvalue().decode("utf-8")))
                except Exception as e: st.error(f"Error parsing file '{uf.name}': {e}")
        
        if not reports_from_json_sources and not alert_text_to_process_str:
            status_placeholder.warning("No data provided for processing.")
        else:
            total_processed_ok = 0; total_errors = 0; processing_details = []
            
            for loaded_data in reports_from_json_sources:
                items_to_process = loaded_data if isinstance(loaded_data, list) else [loaded_data]
                for report_dict in items_to_process:
                    if not isinstance(report_dict, dict):
                        processing_details.append("Skipped invalid non-dictionary item in JSON input.")
                        total_errors += 1; continue
                    
                    api_response = make_api_request("POST", "/api/v1/ingest", payload=report_dict)
                    if api_response and api_response.get('status', '').lower() == 'success':
                        total_processed_ok += 1
                        processing_details.append(f"EIDO JSON (ID: {api_response.get('message_id','N/A')[:15]}...): Processed. Incident: {api_response.get('incident_id','N/A')[:8]}")
                    elif api_response:
                        total_errors += 1
                        processing_details.append(f"EIDO JSON (ID: {api_response.get('message_id','N/A')[:15]}...): {api_response.get('status', 'Failed')}")
                    else: # make_api_request handles st.error display for failures
                        total_errors += 1
                        processing_details.append(f"EIDO JSON: API call failed (check logs/error above).")
            
            st.session_state.last_processed_alert_results = []
            if alert_text_to_process_str:
                api_response_alert = make_api_request("POST", "/api/v1/ingest_alert", payload={"alert_text": alert_text_to_process_str})
                if api_response_alert and api_response_alert.get('details'):
                    alert_results_list = api_response_alert['details']
                    st.session_state.last_processed_alert_results = alert_results_list
                    for idx, res_dict in enumerate(alert_results_list):
                        event_label = f"Event {idx+1}/{len(alert_results_list)}"
                        status_msg = res_dict.get('status', 'Unknown'); msg_id_alert = res_dict.get('message_id', f'alert_event_{idx+1}')[:15]
                        if status_msg.lower() == 'success':
                            total_processed_ok += 1
                            processing_details.append(f"Alert Text ({event_label}, {msg_id_alert}...): Processed. Incident: {res_dict.get('incident_id','N/A')[:8]}")
                        else:
                            total_errors += 1
                            processing_details.append(f"Alert Text ({event_label}, {msg_id_alert}...): {status_msg}")
                elif api_response_alert: # Response but no 'details'
                     total_errors +=1; processing_details.append(f"Alert Text: API response malformed: {str(api_response_alert)[:100]}")
                else: # API call failed entirely
                     total_errors +=1; processing_details.append("Alert Text: API call failed (check logs/error above).")

            if total_errors > 0: status_placeholder.warning(f"Processed: {total_processed_ok} OK, {total_errors} failed.")
            else: status_placeholder.success(f"Processed: {total_processed_ok} inputs successfully.")
            st.session_state.processing_details_to_show = processing_details
            update_dashboard_metrics_and_cache(); get_captured_logs()
            st.session_state.clear_inputs_on_rerun = True; 
            st.session_state.ocr_text_output = "" 
            time.sleep(0.1); st.rerun()

st.sidebar.divider()
with st.sidebar.expander("Admin Actions", expanded=False):
    if st.button("Clear All Incidents from DB", key="clear_all_inc_btn", use_container_width=True):
        api_response_clear = make_api_request("DELETE", "/api/v1/admin/clear_store")
        if api_response_clear:
            st.success(api_response_clear.get("message", "Store cleared."))
        # else: st.error("Failed to clear store via API.") # make_api_request handles error display

        keys_to_reset = ['filtered_incidents_cache', 'active_filters', 'last_processed_alert_results', 'ocr_text_output', 'all_incidents_from_api']
        for key in keys_to_reset:
             if key in st.session_state:
                 if isinstance(st.session_state[key], list): st.session_state[key] = []
                 elif isinstance(st.session_state[key], dict): st.session_state[key] = {}
                 else: init_session_state() # Re-init for safety
        update_dashboard_metrics_and_cache(); get_captured_logs(); time.sleep(0.1); st.rerun()
st.sidebar.divider()
with st.sidebar.expander("Processing Log", expanded=False):
    get_captured_logs()
    log_container = st.container(height=300)
    with log_container:
        if st.session_state.log_messages: st.code("\n".join(st.session_state.log_messages), language='log')
        else: st.caption("Log is empty or no new messages.")

# --- Main Dashboard Content ---
col_title, col_logo_main = st.columns([0.85, 0.15]) 
with col_title: st.title("EIDO Sentinel Demo Dashboard"); st.caption(f"Interactive Application (v0.9.1)")
st.divider()

if 'processing_details_to_show' in st.session_state and st.session_state.processing_details_to_show:
    with st.expander("Last Processing Run Details", expanded=True):
        for detail_msg in st.session_state.processing_details_to_show:
            if "Processed. Incident:" in detail_msg: st.info(detail_msg)
            elif "Skipped" in detail_msg or "Agent returned no results" in detail_msg: st.warning(detail_msg)
            else: st.error(detail_msg)
    del st.session_state.processing_details_to_show 

update_dashboard_metrics_and_cache() # This will now call fetch_all_incidents_from_api()

metric_cols = st.columns(4)
metric_cols[0].metric("Total Incidents", st.session_state.total_incidents)
metric_cols[1].metric("Active Incidents", st.session_state.active_incidents)

# Calculate Avg Reports/Incident based on API fetched data
all_incs_metric = st.session_state.all_incidents_from_api
report_counts_metric = [len(inc.reports_core_data) for inc in all_incs_metric if inc.reports_core_data]
avg_reports_val_metric = sum(report_counts_metric) / len(report_counts_metric) if report_counts_metric else 0
metric_cols[2].metric("Avg Reports/Incident", f"{avg_reports_val_metric:.1f}")
geo_perc = (st.session_state.reports_with_geo / st.session_state.total_reports_geo_checked * 100) if st.session_state.total_reports_geo_checked > 0 else 0
metric_cols[3].metric("Reports w/ Coords", f"{st.session_state.reports_with_geo}/{st.session_state.total_reports_geo_checked}", f"{geo_perc:.0f}%")
st.divider()

st.subheader("Filter & Analyze Incidents")
filter_col1, filter_col2, filter_col3 = st.columns([0.4, 0.3, 0.3])
all_incidents_for_options = st.session_state.all_incidents_from_api # Use API data for filter options
available_types = sorted(list(set(inc.incident_type for inc in all_incidents_for_options if inc.incident_type)))
available_statuses = sorted(list(set(inc.status for inc in all_incidents_for_options if inc.status)))
available_zips = sorted(list(set(zip_code for inc in all_incidents_for_options for zip_code in inc.zip_codes if zip_code)))

def update_filters_ui(): # Renamed to avoid conflict if any other update_filters exists
     st.session_state.active_filters['types'] = st.session_state.filter_type_ms
     st.session_state.active_filters['statuses'] = st.session_state.filter_status_ms
     st.session_state.active_filters['zips'] = st.session_state.filter_zip_ms
     # No rerun needed here, update_dashboard_metrics_and_cache will be called before display

with filter_col1: st.multiselect("Filter by Type:", options=available_types, default=st.session_state.active_filters.get('types',[]), key="filter_type_ms", on_change=update_filters_ui)
with filter_col2: st.multiselect("Filter by Status:", options=available_statuses, default=st.session_state.active_filters.get('statuses',[]), key="filter_status_ms", on_change=update_filters_ui)
with filter_col3: st.multiselect("Filter by ZIP Code:", options=available_zips, default=st.session_state.active_filters.get('zips',[]), key="filter_zip_ms", on_change=update_filters_ui)
st.divider()

# Tabs remain largely the same, but their data source (filtered_incidents_to_display) is now from API
tab_list, tab_map, tab_charts, tab_details, tab_warning, tab_eido_explorer, tab_eido_generator, tab_tutorial = st.tabs([
    "List", "Map", "Charts", "Details", "Warnings", "EIDO Explorer", "EIDO Generator", "Tutorial & Roadmap"
])

filtered_incidents_to_display = st.session_state.filtered_incidents_cache

with tab_list: # Logic remains similar, using filtered_incidents_to_display
    st.caption(f"Displaying {len(filtered_incidents_to_display)} incidents based on filters.")
    if filtered_incidents_to_display:
        list_data = []
        for inc_obj in filtered_incidents_to_display: # inc_obj is PydanticIncident
             list_data.append({"ID": inc_obj.incident_id[:8], "Type": inc_obj.incident_type or "N/A", "Status": inc_obj.status or "N/A",
                               "Last Update": pd.to_datetime(inc_obj.last_updated_at, errors='coerce', utc=True),
                               "Reports": inc_obj.trend_data.get('report_count', len(inc_obj.reports_core_data)), # Use actual if trend not there
                               "Locations": len(inc_obj.locations), "ZIPs": ", ".join(inc_obj.zip_codes or []) or "N/A",
                               "Summary": inc_obj.summary or "N/A"})
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
    else: st.info("No incidents match the current filter criteria, or API data not loaded.")

with tab_map: # Logic remains similar
    st.caption(f"Displaying locations for {len(filtered_incidents_to_display)} filtered incidents.")
    map_points_filtered = []
    for inc_obj in filtered_incidents_to_display: # inc_obj is PydanticIncident
        if inc_obj.locations:
            for lat, lon in inc_obj.locations:
                 if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and -90 <= lat <= 90 and -180 <= lon <= 180:
                    map_points_filtered.append({'lat': lat, 'lon': lon, 'incident_id': inc_obj.incident_id[:8], 'type': inc_obj.incident_type or "Unknown", 'status': inc_obj.status or "Unknown"})

    if map_points_filtered:
        map_df_filtered = pd.DataFrame(map_points_filtered)
        map_df_filtered.drop_duplicates(subset=['lat', 'lon', 'incident_id'], inplace=True) 
        try:
            mid_lat = map_df_filtered['lat'].median(); mid_lon = map_df_filtered['lon'].median()
            view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=10, pitch=45)
            scatter_layer = pdk.Layer('ScatterplotLayer', data=map_df_filtered, get_position='[lon, lat]', get_fill_color='[0, 188, 212, 180]', get_line_color='[0,100,120,255]', get_radius=50, pickable=True, auto_highlight=True, lineWidthMinPixels=1)
            heatmap_layer = pdk.Layer("HeatmapLayer", data=map_df_filtered, opacity=0.6, get_position=["lon", "lat"], aggregation=pdk.types.String("MEAN"), threshold=0.05, get_weight=1, pickable=False, color_range=[[255,255,204,20],[255,237,160,80],[254,217,118,120],[254,178,76,160],[253,141,60,200],[227,26,28,255]])
            tooltip = {"html": "<b>Incident:</b> {incident_id}<br/><b>Type:</b> {type}<br/><b>Status:</b> {status}", "style": {"backgroundColor": "var(--streamlit-primary-color)", "color": "white", "border-radius": "5px", "padding": "5px"}}
            st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v10', initial_view_state=view_state, layers=[heatmap_layer, scatter_layer], tooltip=tooltip))
            with st.expander("Show Raw Map Data"): st.dataframe(map_df_filtered[['incident_id', 'type', 'status', 'lat', 'lon']].round(6), use_container_width=True, hide_index=True)
        except Exception as map_error: st.error(f"Error displaying PyDeck map: {map_error}")
    else: st.info("No geocoded locations match the current filter criteria, or API data not loaded.")

with tab_charts: # Logic remains similar
    st.caption(f"Displaying charts for {len(filtered_incidents_to_display)} filtered incidents.")
    if filtered_incidents_to_display:
        chart_data = [{"Status": inc.status or "N/A", "Type": inc.incident_type or "N/A"} for inc in filtered_incidents_to_display]
        df_chart = pd.DataFrame(chart_data)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### Status Distribution")
            status_counts = df_chart['Status'].value_counts()
            if not status_counts.empty: st.bar_chart(status_counts, color="#00BCD4") 
            else: st.caption("No status data.")
        with chart_col2:
            st.markdown("##### Top Incident Types")
            type_counts = df_chart['Type'].value_counts().head(10)
            if not type_counts.empty: st.bar_chart(type_counts, color="#FFDA63") 
            else: st.caption("No type data.")
    else: st.info("No incidents match filters to display charts, or API data not loaded.")

with tab_details: # Logic remains similar, but incident fetching for detail can be direct
    st.caption(f"Select one of the {len(filtered_incidents_to_display)} filtered incidents for details.")
    if filtered_incidents_to_display:
        details_map = {f"{inc.incident_id[:8]} - {inc.incident_type or 'N/A'} ({inc.status})": inc.incident_id for inc in filtered_incidents_to_display}
        details_options = ["-- Select Incident --"] + list(details_map.keys())
        selected_label = st.selectbox("Select Filtered Incident:", options=details_options, index=0, key="detail_incident_selector")

        if selected_label != "-- Select Incident --":
            selected_full_id = details_map.get(selected_label)
            # Find the selected PydanticIncident from the already fetched list
            selected_incident: Optional[PydanticIncident] = next((inc for inc in st.session_state.all_incidents_from_api if inc.incident_id == selected_full_id), None)

            if selected_incident:
                st.markdown(f"#### Incident `{selected_incident.incident_id[:8]}` Details"); st.divider()
                info_col1, info_col2 = st.columns(2)
                with info_col1: # Displaying data from PydanticIncident object
                    st.markdown(f"**Type:** {selected_incident.incident_type or 'N/A'}")
                    st.markdown(f"**Status:** `{selected_incident.status or 'Unknown'}`")
                    st.markdown(f"**Reports:** {selected_incident.trend_data.get('report_count', len(selected_incident.reports_core_data))}")
                with info_col2:
                    created_dt = pd.to_datetime(selected_incident.created_at, utc=True, errors='coerce')
                    updated_dt = pd.to_datetime(selected_incident.last_updated_at, utc=True, errors='coerce')
                    st.markdown(f"**Created:** {created_dt.strftime('%Y-%m-%d %H:%M Z') if pd.notna(created_dt) else 'N/A'}")
                    st.markdown(f"**Updated:** {updated_dt.strftime('%Y-%m-%d %H:%M Z') if pd.notna(updated_dt) else 'N/A'}")
                    st.markdown(f"**ZIPs:** `{', '.join(selected_incident.zip_codes) or 'N/A'}`")

                st.divider(); st.markdown("##### AI Summary"); st.info(selected_incident.summary or "_No summary generated._")
                st.markdown("##### Recommended Actions")
                if selected_incident.recommended_actions: st.markdown("\n".join(f"- {action}" for action in selected_incident.recommended_actions))
                else: st.caption("_No actions recommended._")
                st.divider()

                with st.expander(f"Associated Reports ({len(selected_incident.reports_core_data or [])})", expanded=False):
                     if selected_incident.reports_core_data:
                         sorted_reports_core = sorted(selected_incident.reports_core_data, key=lambda r: r.timestamp if r.timestamp else datetime.min.replace(tzinfo=timezone.utc), reverse=True)
                         report_data_list = []
                         for r_core in sorted_reports_core:
                             ts = pd.to_datetime(r_core.timestamp, errors='coerce', utc=True)
                             report_data_list.append({
                                 "Rept ID": r_core.report_id[:8], "Timestamp": ts,
                                 "Ext. ID": r_core.external_incident_id or "N/A", "Source": r_core.source or "N/A",
                                 "Description": r_core.description or "N/A", "Address": r_core.location_address or "N/A",
                                 "Coords": f"{r_core.coordinates[0]:.5f}, {r_core.coordinates[1]:.5f}" if r_core.coordinates and len(r_core.coordinates) == 2 else "N/A",
                                 "ZIP": r_core.zip_code or "N/A"
                             })
                         df_reports = pd.DataFrame(report_data_list)
                         st.dataframe(df_reports, hide_index=True, use_container_width=True, column_config={"Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss Z")})
                     else: st.info("No report data associated.")
                with st.expander(f"Match/Update History", expanded=False):
                    match_info = selected_incident.trend_data.get('last_match_info', 'N/A')
                    st.caption(f"**Last Update Reason:** {match_info}")
            else: st.warning(f"Could not retrieve details for incident ID {selected_label}.")
    else: st.info("No incidents match filters to show details, or API data not loaded.")

with tab_warning: # Logic remains similar
    st.subheader("Generate Incident Warning Text")
    st.caption(f"Based on the {len(filtered_incidents_to_display)} currently filtered incidents.")
    if filtered_incidents_to_display:
        warning_level = st.select_slider("Select Warning Severity:", options=["Informational", "Advisory", "Watch", "Warning"], value="Advisory", key="warn_level")
        custom_message = st.text_area("Add Custom Message (Optional):", height=100, key="warn_custom_msg")
        if st.button("Generate Warning", use_container_width=True, key="warn_generate_btn"):
            warning_text = f"**--- {warning_level.upper()} ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M Z')}) ---**\n\n"
            if custom_message: warning_text += f"{custom_message}\n\n"
            warning_text += f"**Summary of Filtered Incidents ({len(filtered_incidents_to_display)}):**\n"
            for inc_obj in filtered_incidents_to_display[:10]: 
                warning_text += f"- **ID:** {inc_obj.incident_id[:8]}, **Type:** {inc_obj.incident_type or 'N/A'}, **Status:** {inc_obj.status or 'N/A'}\n"
                loc_summary = ", ".join(inc_obj.addresses[:1]) or (f"Coords: {inc_obj.locations[0]}" if inc_obj.locations else "N/A Location")
                warning_text += f"  Location: {loc_summary}\n"
                warning_text += f"  Summary: {inc_obj.summary[:150]}...\n"
            if len(filtered_incidents_to_display) > 10: warning_text += f"\n... and {len(filtered_incidents_to_display) - 10} more incidents matching filters."
            st.text_area("Generated Warning Text (Copy below):", value=warning_text, height=300, key="warning_output_area")
    else: st.info("Apply filters to select incidents for warning generation, or API data not loaded.")

with tab_eido_explorer: # Logic remains similar
    st.subheader("Explore Original/Generated EIDO Data")
    st.caption("View the EIDO JSON associated with processed reports (original or LLM-generated).")
    if filtered_incidents_to_display:
        explorer_map = {f"{inc.incident_id[:8]} - {inc.incident_type or 'N/A'} ({inc.status})": inc.incident_id for inc in filtered_incidents_to_display}
        explorer_options = ["-- Select Incident --"] + list(explorer_map.keys())
        selected_inc_label_exp = st.selectbox("Select Incident to Explore:", options=explorer_options, index=0, key="eido_explorer_inc_select")

        if selected_inc_label_exp != "-- Select Incident --":
            selected_inc_full_id_exp = explorer_map.get(selected_inc_label_exp)
            selected_inc_obj_exp: Optional[PydanticIncident] = next((inc for inc in st.session_state.all_incidents_from_api if inc.incident_id == selected_inc_full_id_exp), None)


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
                                st_ace(value=eido_str_display, language="json", theme="tomorrow_night_blue", readonly=True, key=f"ace_editor_exp_{selected_report_id_exp}", height=400, wrap=True) 
                                st.download_button(label="Download this EIDO JSON", data=eido_str_display.encode('utf-8'), file_name=f"eido_report_{selected_report_core_exp.report_id[:8]}.json", mime="application/json", key=f"dl_eido_report_exp_{selected_report_id_exp}")
                            except Exception as json_err: st.error(f"Error formatting EIDO JSON for display: {json_err}"); st.json(eido_to_display) 
                        elif selected_report_core_exp: st.warning("Selected report does not have stored EIDO data.")
            elif selected_inc_obj_exp: st.info("Selected incident has no associated report data.")
    else: st.info("No incidents match filters to explore EIDO data, or API data not loaded.")

with tab_eido_generator: 
    st.subheader("Generate Compliant EIDO JSON"); st.info("Use LLM assistance to fill a standard EIDO template based on a scenario description.")
    available_templates = list_files_in_dir(TEMPLATE_DIR) # Local file listing
    if not available_templates:
        st.warning(f"No EIDO templates found in `{TEMPLATE_DIR}`. Create JSON templates with placeholders like `[PLACEHOLDER]`.")
    else:
        template_options = ["-- Select Template --"] + available_templates
        selected_template_file = st.selectbox("Select EIDO Template:", options=template_options, index=0, key="generator_template_select")
        scenario_description = st.text_area("Enter Scenario Description:", height=150, key="generator_scenario_input", placeholder="Describe the incident (e.g., 'Structure fire at 100 Main St, Apt 5, reported by Engine 3 at 08:15 UTC...')")

        if st.button("Generate EIDO from Template via API", key="generator_button_api", disabled=(selected_template_file == "-- Select Template --" or not scenario_description)):
            with st.spinner("Generating EIDO JSON via API..."):
                payload = {"template_name": selected_template_file, "scenario_description": scenario_description}
                api_response_gen = make_api_request("POST", "/api/v1/generate_eido_from_template", payload=payload)
                
                if api_response_gen and "generated_eido" in api_response_gen:
                    # API returns parsed JSON, so dump it back to string for display/download if needed
                    st.session_state.generated_eido_json = json.dumps(api_response_gen["generated_eido"], indent=2)
                    st.success("EIDO JSON generated successfully via API!")
                else:
                    st.session_state.generated_eido_json = None
                    st.error("Failed to generate EIDO JSON from template using API. Check logs/error above.")
                get_captured_logs()

        if st.session_state.generated_eido_json:
            st.markdown("---"); st.markdown("**Generated EIDO JSON:**")
            try:
                # generated_eido_json is already a string from the API success path
                st_ace(value=st.session_state.generated_eido_json, language="json", theme="tomorrow_night_blue", keybinding="vscode", font_size=12, height=400, show_gutter=True, show_print_margin=False, wrap=True, auto_update=False, readonly=True, key="ace_editor_generator_output") 
                st.download_button(label="Download Generated EIDO", data=st.session_state.generated_eido_json.encode('utf-8'), file_name=f"generated_{selected_template_file.replace('.json','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", mime="application/json", key="dl_generated_eido")
            except Exception as display_err: st.error(f"Error displaying/formatting generated EIDO: {display_err}"); st.text_area("Raw Generated Output:", value=st.session_state.generated_eido_json, height=400)

with tab_tutorial: # Logic remains same
    st.header("EIDO Sentinel Demo Application Tutorial")
    st.markdown(f"""
    Welcome to the EIDO Sentinel interactive demo! This guide will walk you through using the application's key features. 
    The application sidebar uses a dark theme, while the main content area uses a light theme for clarity.
    You can visit the main project showcase at [EIDO Sentinel Landing Page]({LANDING_PAGE_URL}).
    """) 
    st.subheader("1. Agent Status (Sidebar)")
    st.markdown("""
    - **Backend API URL:** Shows the configured URL for the FastAPI backend. All data processing and LLM interactions happen there.
    - Ensure the backend is running and accessible from this Streamlit application.
    """)

    st.subheader("2. Ingesting Data (Sidebar)")
    st.markdown("""
    Input data into EIDO Sentinel via the **Data Ingestion** section in the sidebar. Processing requests are sent to the backend API.
    - **EIDO JSON Tab:** Upload files, paste JSON, or load samples.
    - **Raw Text Tab:** Input unstructured text. The backend agent will parse this.
    - **Image (OCR) Tab:** Upload an image for text extraction. The extracted text populates the 'Raw Text' area for processing.
    
    Click **Process Inputs** to send data to the backend. Results will update the dashboard.
    """)
    st.subheader("3. Understanding the Dashboard Tabs")
    st.markdown(f"""
    The main dashboard visualizes data retrieved from the backend API:
    - **Metrics Bar & Filters:** Overview stats and filtering options.
    - **List, Map, Charts, Details Tabs:** Various views of incident data.
    - **Warnings Tab:** Generate warning text based on filtered incidents.
    - **EIDO Explorer Tab:** View original/generated EIDO for reports within incidents.
    - **EIDO Generator Tab:** Use backend LLM assistance to fill EIDO templates.
    - **Tutorial & Roadmap Tab:** This guide and project future plans. Visit the [Landing Page]({LANDING_PAGE_URL}) for more.
    """)
    st.info("Remember: This is a Proof-of-Concept. LLM outputs (via backend) can vary.")

st.divider()
st.caption(f"EIDO Sentinel v0.9.1 | Interactive Demo Application | API: {st.session_state.api_base_url}")
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; padding-bottom: 20px; font-size: 0.9em;">
        <a href="https://www.linkedin.com/in/yourprofile" target="_blank" style="margin: 0 10px;">LinkedIn</a> | 
        <a href="https://github.com/LXString/eido-sentinel" target="_blank" style="margin: 0 10px;">GitHub</a> | 
        <a href="https://www.sdsc.edu" target="_blank" style="margin: 0 10px;">UCSD SDSC</a>
    </div>
    """, unsafe_allow_html=True)
get_captured_logs()