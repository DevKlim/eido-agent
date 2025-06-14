import streamlit as st
import json
import os
import pandas as pd
import time
from datetime import datetime, timezone
import sys
import logging
from io import StringIO, BytesIO
import pydeck as pdk
from streamlit_ace import st_ace
from typing import List, Dict, Optional, Any
import requests

# --- Setup Python Path & Imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

modules_imported_successfully = True
import_error_message = ""
original_error = None
try:
    from config.settings import settings as local_settings
    from data_models.schemas import Incident as PydanticIncident
    from utils.ocr_processor import extract_text_from_image
except Exception as e:
    modules_imported_successfully = False
    import_error_message = f"Setup Error: {e}"
    original_error = e

# --- Logging Setup ---
# Setup logging regardless of other imports to capture errors
log_level_to_set = getattr(local_settings, 'log_level', 'INFO').upper() if 'local_settings' in locals() else 'INFO'
logging.basicConfig(level=log_level_to_set, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
logger_ui = logging.getLogger("EidoSentinelUI")
log_stream = StringIO()
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
stream_handler_ui = logging.StreamHandler(log_stream)
stream_handler_ui.setFormatter(log_formatter)
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == log_stream for h in root_logger.handlers):
    root_logger.addHandler(stream_handler_ui)
    logger_ui.debug("UI Log Capture StreamHandler added to root logger.")

# --- Environment and API Configuration ---
# This is the core logic for deployment awareness.
# We check for st.secrets, which only exists in Streamlit Cloud deployments.
IS_DEPLOYED = hasattr(st, 'secrets') and 'API_BASE_URL' in st.secrets

if IS_DEPLOYED:
    # --- DEPLOYED MODE ---
    # The app is running on Streamlit Cloud.
    # It reads the API_BASE_URL from the secrets you provide in the dashboard.
    API_BASE_URL = st.secrets['API_BASE_URL']
    logger_ui.info(f"Running in DEPLOYED mode. API URL from secrets: {API_BASE_URL}")
else:
    # --- LOCAL DEVELOPMENT MODE ---
    # The app is running on your local machine.
    # It falls back to the settings loaded from your local .env file.
    API_BASE_URL = local_settings.api_base_url if modules_imported_successfully else "http://localhost:8000"
    logger_ui.info(f"Running in LOCAL mode. API URL from settings.py: {API_BASE_URL}")

LANDING_PAGE_URL = API_BASE_URL  # The backend API also serves the landing page.

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="EIDO Sentinel | AI Incident Processor",
    page_icon="img/logo_icon_light.png",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': LANDING_PAGE_URL,
        'Report a bug': "https://github.com/DevKlim/eido-agent/issues",
        'About': f"# EIDO Sentinel v0.9.1\nAI-Powered Emergency Incident Processor. Visit our showcase at {LANDING_PAGE_URL}"
    }
)

if not modules_imported_successfully:
    st.error(f"CRITICAL ERROR: Failed during application setup. Please check the terminal logs.")
    st.error(f"Details: {import_error_message}")
    if original_error:
        st.exception(original_error)
    st.warning("Ensure all dependencies are installed (`pip install -r requirements.txt`) and required services (like Tesseract OCR) are available.")
    st.stop()

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'log_messages': [], 'map_data': pd.DataFrame(columns=['lat', 'lon']),
        'total_incidents': 0, 'active_incidents': 0,
        'clear_inputs_on_rerun': False, 'generated_eido_json': None,
        'filtered_incidents_cache': [], 'active_filters': {},
        'ocr_text_output': "", 'all_incidents_from_api': [],
        'local_geocoded_locations': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    
    st.session_state.api_base_url = API_BASE_URL

init_session_state()

# --- API Helper Functions ---
def make_api_request(method: str, endpoint: str, payload: Optional[Dict] = None, params: Optional[Dict] = None) -> Optional[Any]:
    url = f"{st.session_state.api_base_url}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=payload, timeout=60)
        elif method.upper() == "PUT":
            response = requests.put(url, json=payload, timeout=30)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status()
        if response.status_code == 204:  # No Content
            return True
        return response.json() if response.content else True
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error ({e.response.status_code}): {e.response.text}")
        logger_ui.error(f"API HTTP Error for {url}: {e.response.status_code} - {e.response.text}", exc_info=True)
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: Could not connect to {st.session_state.api_base_url}. Is the backend running?")
        logger_ui.error(f"API Connection Error for {url}: {e}", exc_info=True)
    return None

# --- UI Helper Functions ---
def get_captured_logs():
     log_stream.seek(0)
     logs_captured_this_run = log_stream.read()
     log_stream.truncate(0)
     log_stream.seek(0)
     new_entries = [entry for entry in logs_captured_this_run.strip().split('\n') if entry.strip()]
     if new_entries:
         st.session_state.log_messages = new_entries + st.session_state.log_messages[:199]

def fetch_all_incidents_from_api():
    data = make_api_request("GET", "/api/v1/incidents")
    if data and isinstance(data, list):
        try:
            st.session_state.all_incidents_from_api = [PydanticIncident(**inc) for inc in data]
        except Exception as e:
            st.error(f"Error parsing incidents from API: {e}")
            st.session_state.all_incidents_from_api = []
    else:
        st.session_state.all_incidents_from_api = []

def update_dashboard_metrics_and_cache():
    fetch_all_incidents_from_api()
    all_incidents = st.session_state.all_incidents_from_api
    st.session_state.total_incidents = len(all_incidents)
    active_statuses = ["active", "updated", "monitoring"]
    st.session_state.active_incidents = sum(1 for inc in all_incidents if inc.status and inc.status.lower() in active_statuses)
    
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

def list_files_in_dir(dir_path, extension=".json"):
    if not os.path.exists(dir_path): return []
    return sorted([f for f in os.listdir(dir_path) if f.endswith(extension)])

# --- Load static assets ---
UI_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(UI_DIR, '..', 'static', 'images', 'logo_icon_dark.png')
CUSTOM_CSS_PATH = os.path.join(UI_DIR, 'custom_styles.css')
with open(CUSTOM_CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image(LOGO_PATH, width=60)
st.sidebar.markdown(f"**[Project Showcase]({LANDING_PAGE_URL})**")
st.sidebar.caption("AI Incident Processor")
st.sidebar.divider()
st.sidebar.header("Agent Status")
st.sidebar.info(f"Backend API: {st.session_state.api_base_url}")
st.sidebar.divider()
st.sidebar.header("Data Ingestion")

# Clear inputs if needed
if st.session_state.get('clear_inputs_on_rerun', False):
    st.session_state.json_input_area_val = ""
    st.session_state.alert_text_input_area_val = ""
    st.session_state.ocr_text_output = ""
    st.session_state.clear_inputs_on_rerun = False

ingest_tab1, ingest_tab2, ingest_tab3 = st.sidebar.tabs(["EIDO JSON", "Raw Text", "Image (OCR)"])
with ingest_tab1:
    uploaded_files = st.file_uploader("Upload EIDO JSON File(s)", type="json", accept_multiple_files=True, key="file_uploader_key")
    json_input_area = st.text_area("Paste EIDO JSON", key="json_input_area_val", height=150)
    sample_dir = os.path.join(PROJECT_ROOT, 'sample_eido')
    selected_sample = st.selectbox("Or Load Sample EIDO:", options=["-- Select Sample --"] + list_files_in_dir(sample_dir), key="sample_select_key")
with ingest_tab2:
    alert_text_input_area = st.text_area("Paste Raw Alert Text", key="alert_text_input_area_val", height=200)
with ingest_tab3:
    uploaded_image_ocr = st.file_uploader("Upload Image for OCR", type=["png", "jpg", "jpeg"])
    if uploaded_image_ocr:
        if st.button("Extract Text from Image", key="ocr_extract_button"):
            with st.spinner("Performing OCR..."):
                ocr_text = extract_text_from_image(BytesIO(uploaded_image_ocr.getvalue()))
                if ocr_text:
                    st.session_state.alert_text_input_area_val = ocr_text
                    st.success("OCR successful! Text placed in 'Raw Alert Text' tab.")
                else:
                    st.error("OCR failed. Is Tesseract installed and in PATH?")
                get_captured_logs()

if st.sidebar.button("Process Inputs", type="primary", use_container_width=True):
    status_placeholder = st.sidebar.empty()
    status_placeholder.info("Sending inputs to backend for processing...")
    
    with st.spinner('Agent is processing...'):
        # Process JSON
        json_to_process = []
        if st.session_state.json_input_area_val:
            try: json_to_process.append(json.loads(st.session_state.json_input_area_val))
            except json.JSONDecodeError: st.error("Pasted JSON is invalid.")
        if st.session_state.sample_select_key != "-- Select Sample --":
            with open(os.path.join(sample_dir, st.session_state.sample_select_key), 'r') as f:
                json_to_process.append(json.load(f))
        for uf in uploaded_files:
            json_to_process.append(json.loads(uf.getvalue()))

        for item in json_to_process:
            make_api_request("POST", "/api/v1/ingest", payload=item)
        
        # Process Raw Text
        if st.session_state.alert_text_input_area_val:
            make_api_request("POST", "/api/v1/ingest_alert", payload={"alert_text": st.session_state.alert_text_input_area_val})

    status_placeholder.success("Processing complete!")
    st.session_state.clear_inputs_on_rerun = True
    update_dashboard_metrics_and_cache()
    get_captured_logs()
    time.sleep(1)
    st.rerun()

st.sidebar.divider()
with st.sidebar.expander("Admin Actions"):
    if st.button("Clear All Incidents", use_container_width=True):
        if make_api_request("DELETE", "/api/v1/admin/clear_store"):
            st.success("Incident store cleared.")
            st.session_state.all_incidents_from_api = []
            st.session_state.filtered_incidents_cache = []
            update_dashboard_metrics_and_cache()
            time.sleep(1)
            st.rerun()

with st.sidebar.expander("Processing Log", expanded=False):
    get_captured_logs()
    st.code("\n".join(st.session_state.log_messages), language='log')

# --- Main Dashboard ---
st.title("EIDO Sentinel Dashboard")
st.caption(f"v0.9.1 | Connected to: {API_BASE_URL}")
st.divider()

update_dashboard_metrics_and_cache()
metric_cols = st.columns(3)
metric_cols[0].metric("Total Incidents", st.session_state.total_incidents)
metric_cols[1].metric("Active Incidents", st.session_state.active_incidents)
report_counts = [len(inc.reports_core_data) for inc in st.session_state.all_incidents_from_api]
avg_reports = sum(report_counts) / len(report_counts) if report_counts else 0
metric_cols[2].metric("Avg Reports/Incident", f"{avg_reports:.1f}")
st.divider()

st.subheader("Incident Data")

if not st.session_state.all_incidents_from_api:
    st.info("No incident data loaded. Please ingest data using the sidebar.")
    st.stop()

# --- Filtering ---
filter_col1, filter_col2, filter_col3 = st.columns([0.4, 0.3, 0.3])
all_incidents = st.session_state.all_incidents_from_api
available_types = sorted(list(set(inc.incident_type for inc in all_incidents if inc.incident_type)))
available_statuses = sorted(list(set(inc.status for inc in all_incidents if inc.status)))
available_zips = sorted(list(set(zip_code for inc in all_incidents for zip_code in inc.zip_codes)))

def update_filters():
    st.session_state.active_filters['types'] = st.session_state.get('filter_type_ms', [])
    st.session_state.active_filters['statuses'] = st.session_state.get('filter_status_ms', [])
    st.session_state.active_filters['zips'] = st.session_state.get('filter_zip_ms', [])

with filter_col1:
    st.multiselect("Filter by Type:", options=available_types, key="filter_type_ms", on_change=update_filters)
with filter_col2:
    st.multiselect("Filter by Status:", options=available_statuses, key="filter_status_ms", on_change=update_filters)
with filter_col3:
    st.multiselect("Filter by ZIP Code:", options=available_zips, key="filter_zip_ms", on_change=update_filters)
st.divider()

filtered_incidents = st.session_state.filtered_incidents_cache

# --- TABS ---
tab_list, tab_map, tab_details, tab_tools = st.tabs(["List", "Map", "Details", "Tools"])

with tab_list:
    st.caption(f"Displaying {len(filtered_incidents)} incidents based on filters.")
    if filtered_incidents:
        df_list = pd.DataFrame([{
            "ID": inc.incident_id[:8], "Type": inc.incident_type, "Status": inc.status,
            "Last Update": inc.last_updated_at, "Reports": len(inc.reports_core_data),
            "Summary": inc.summary[:100] + '...' if inc.summary else ''
        } for inc in filtered_incidents])
        st.dataframe(df_list, use_container_width=True, hide_index=True)

with tab_map:
    map_points = []
    for inc in filtered_incidents:
        if inc.locations:
            for lat, lon in inc.locations:
                map_points.append({'lat': lat, 'lon': lon, 'tooltip': f"ID: {inc.incident_id[:8]}\nType: {inc.incident_type}"})
    if map_points:
        df_map = pd.DataFrame(map_points)
        view_state = pdk.ViewState(latitude=df_map['lat'].mean(), longitude=df_map['lon'].mean(), zoom=11, pitch=45)
        layer = pdk.Layer('ScatterplotLayer', data=df_map, get_position='[lon, lat]', get_color='[200, 30, 0, 160]', get_radius=100, pickable=True)
        st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v10', initial_view_state=view_state, layers=[layer], tooltip={"text": "{tooltip}"}))
    else:
        st.info("No geocoded locations to display for the current filter.")

with tab_details:
    if filtered_incidents:
        options = {f"{inc.incident_id[:8]} - {inc.incident_type}": inc for inc in filtered_incidents}
        selected_key = st.selectbox("Select Incident:", options=list(options.keys()))
        if selected_key:
            selected_incident = options[selected_key]
            st.subheader(f"Incident: {selected_incident.incident_id}")
            st.json(selected_incident.model_dump_json(indent=2))

with tab_tools:
    st.subheader("EIDO Generator")
    template_dir = os.path.join(PROJECT_ROOT, 'eido_templates')
    templates = list_files_in_dir(template_dir)
    selected_template = st.selectbox("Select Template", options=templates)
    scenario = st.text_area("Scenario Description")
    if st.button("Generate EIDO"):
        if selected_template and scenario:
            payload = {"template_name": selected_template, "scenario_description": scenario}
            response = make_api_request("POST", "/api/v1/generate_eido_from_template", payload=payload)
            if response and 'generated_eido' in response:
                st.session_state.generated_eido_json = json.dumps(response['generated_eido'], indent=2)
    if st.session_state.get('generated_eido_json'):
        st_ace(value=st.session_state.generated_eido_json, language='json', readonly=True)

st.divider()
st.caption(f"EIDO Sentinel v0.9.1 | End of Dashboard")