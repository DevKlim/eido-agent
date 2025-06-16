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
from PIL import Image
import urllib.parse

# --- Page Configuration ---
# Load the icon as a PIL Image object to ensure the path is always correct.
try:
    PAGE_ICON_PATH = os.path.abspath(os.path.join(os.path.dirname(
        __file__), '..', 'static', 'images', 'logo_icon_light.png'))
    page_icon_img = Image.open(PAGE_ICON_PATH)
except FileNotFoundError:
    page_icon_img = "🤖"  # Fallback to an emoji if the image is not found

st.set_page_config(
    layout="wide",
    page_title="EIDO Sentinel | AI Incident Processor",
    page_icon=page_icon_img,
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': "https://github.com/LXString/eido-sentinel",
        'Report a bug': "https://github.com/LXString/eido-sentinel/issues",
        'About': "# EIDO Sentinel v0.9.1\nAI-Powered Emergency Incident Processor. A dynamic link to the project showcase is in the sidebar."
    }
)

# --- Defensive Setup with Graceful Error Handling ---
# This block prevents the "blank page" crash by catching startup errors.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

modules_imported_successfully = False
import_error_message = ""
original_error = None
local_settings = None  # Initialize local_settings to None

# --- Logging Setup ---
# Setup logging to capture messages for display in the UI
log_stream = StringIO()
log_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
stream_handler_ui = logging.StreamHandler(log_stream)
stream_handler_ui.setFormatter(log_formatter)
root_logger = logging.getLogger()
# Avoid adding handler multiple times on reruns
if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == log_stream for h in root_logger.handlers):
    root_logger.addHandler(stream_handler_ui)

# Set root logger level based on settings, or default to INFO
log_level_to_set = 'INFO'  # Default
try:
    from config.settings import settings as temp_settings
    local_settings = temp_settings  # Assign to local_settings for later use
    log_level_to_set = getattr(local_settings, 'log_level', 'INFO').upper()
    root_logger.setLevel(log_level_to_set)
    # These imports are now safe to perform within the success block
    from data_models.schemas import Incident as PydanticIncident
    from utils.ocr_processor import extract_text_from_image
    modules_imported_successfully = True
except Exception as e:
    import_error_message = f"A required module failed to import. This is often caused by missing dependencies or a misconfigured '.env' file that prevents 'config/settings.py' from loading."
    original_error = e
    # Set to ERROR if settings can't be loaded
    root_logger.setLevel(logging.ERROR)

logger_ui = logging.getLogger("EidoSentinelUI")
if modules_imported_successfully:
    logger_ui.debug("UI Log Capture StreamHandler added to root logger.")


# --- Environment and API Configuration ---
API_BASE_URL = "http://localhost:8000"  # Default fallback
IS_DEPLOYED = False

try:
    if 'API_BASE_URL' in st.secrets:
        API_BASE_URL = st.secrets['API_BASE_URL']
        IS_DEPLOYED = True
        logger_ui.info(
            f"Running in DEPLOYED mode. API URL from secrets: {API_BASE_URL}")
    elif local_settings:  # Use local_settings if successfully loaded
        API_BASE_URL = local_settings.api_base_url
        logger_ui.info(
            f"Running in LOCAL mode. API URL from settings.py: {API_BASE_URL}")
    else:  # Fallback if local_settings could not be loaded
        logger_ui.info(
            f"Running in LOCAL mode (settings not loaded). Default API URL: {API_BASE_URL}")
except (FileNotFoundError, AttributeError):
    # This error is raised when st.secrets is accessed and no secrets file is found,
    # or if local_settings was not successfully loaded and its attribute is accessed.
    if local_settings:
        API_BASE_URL = local_settings.api_base_url
    logger_ui.info(
        f"Running in LOCAL mode (no secrets file or settings error). API URL: {API_BASE_URL}")

# This is now correctly set based on the environment.
LANDING_PAGE_URL = API_BASE_URL

# --- Session State Initialization ---


def init_session_state():
    """Initializes all required session_state keys to prevent KeyErrors."""
    defaults = {
        'log_messages': [], 'map_data': pd.DataFrame(columns=['lat', 'lon']),
        'total_incidents': 0, 'active_incidents': 0,
        'clear_inputs_on_rerun': False, 'generated_eido_json': None,
        'filtered_incidents_cache': [], 'active_filters': {},
        'ocr_text_output': "", 'all_incidents_from_api': [],
        'local_geocoded_locations': {},
        'api_is_reachable': None,  # Track API status: None=unknown, True=ok, False=fail
        'json_input_area_val': "",      # Initialize widget key
        'alert_text_input_area_val': ""  # Initialize widget key
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
            st.session_state.api_is_reachable = False
            return None

        response.raise_for_status()
        st.session_state.api_is_reachable = True
        if response.status_code == 204:  # No Content
            return True
        return response.json() if response.content else True
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error ({e.response.status_code}): {e.response.text}")
        logger_ui.error(
            f"API HTTP Error for {url}: {e.response.status_code} - {e.response.text}", exc_info=True)
        st.session_state.api_is_reachable = False  # Mark as unreachable on HTTP error
    except requests.exceptions.RequestException as e:
        st.error(
            f"API Connection Error: Could not connect to {st.session_state.api_base_url}. Is the backend running?")
        logger_ui.error(f"API Connection Error for {url}: {e}", exc_info=True)
        # Mark as unreachable on connection error
        st.session_state.api_is_reachable = False
    return None

# --- UI Helper Functions ---


def get_captured_logs():
    log_stream.seek(0)
    logs_captured_this_run = log_stream.read()
    log_stream.truncate(0)
    log_stream.seek(0)
    new_entries = [entry for entry in logs_captured_this_run.strip().split(
        '\n') if entry.strip()]
    if new_entries:
        st.session_state.log_messages = new_entries + \
            st.session_state.log_messages[:199]


def fetch_all_incidents_from_api():
    """Fetches all incidents and correctly parses them into Pydantic models."""
    # Only attempt if modules were imported and API is not known to be unreachable
    if not modules_imported_successfully or st.session_state.api_is_reachable is False:
        st.session_state.all_incidents_from_api = []
        return

    data = make_api_request("GET", "/api/v1/incidents")
    if data and isinstance(data, list):
        try:
            # Use the imported PydanticIncident class directly
            st.session_state.all_incidents_from_api = [
                PydanticIncident(**inc) for inc in data]
        except Exception as e:
            st.error(f"Error parsing incidents from API: {e}")
            logger_ui.error(f"Pydantic parsing error: {e}", exc_info=True)
            st.session_state.all_incidents_from_api = []
    # If make_api_request returns None or non-list, it sets api_is_reachable to False.
    # We clear the data to reflect this.
    elif st.session_state.api_is_reachable is False:
        st.session_state.all_incidents_from_api = []


def update_dashboard_metrics_and_cache():
    fetch_all_incidents_from_api()
    all_incidents = st.session_state.all_incidents_from_api
    st.session_state.total_incidents = len(all_incidents)
    active_statuses = ["active", "updated", "monitoring"]
    st.session_state.active_incidents = sum(
        1 for inc in all_incidents if inc.status and inc.status.lower() in active_statuses)

    filtered_inc_list = all_incidents
    active_filters = st.session_state.get('active_filters', {})
    if active_filters.get('types'):
        filtered_inc_list = [
            inc for inc in filtered_inc_list if inc.incident_type in active_filters['types']]
    if active_filters.get('statuses'):
        filtered_inc_list = [
            inc for inc in filtered_inc_list if inc.status in active_filters['statuses']]
    if active_filters.get('zips'):
        filtered_inc_list = [inc for inc in filtered_inc_list if any(
            zip_code in active_filters['zips'] for zip_code in inc.zip_codes)]

    st.session_state.filtered_incidents_cache = sorted(
        filtered_inc_list,
        key=lambda x: x.last_updated_at if x.last_updated_at else datetime.min.replace(
            tzinfo=timezone.utc),
        reverse=True
    )


def list_files_in_dir(dir_path, extension=".json"):
    if not os.path.exists(dir_path):
        return []
    return sorted([f for f in os.listdir(dir_path) if f.endswith(extension)])


# --- Load static assets ---
UI_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(UI_DIR, '..', 'static',
                         'images', 'logo_icon_dark.png')
CUSTOM_CSS_PATH = os.path.join(UI_DIR, 'custom_styles.css')
with open(CUSTOM_CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar (Initial Render) ---
st.sidebar.image(LOGO_PATH, width=60)
st.sidebar.markdown(f"**[Project Showcase]({LANDING_PAGE_URL})**")
st.sidebar.caption("AI Incident Processor")
st.sidebar.divider()

# --- Main Dashboard (Initial Render) ---
st.title("EIDO Sentinel Dashboard")
st.caption(f"v0.9.1 | Connected to: {API_BASE_URL}")
st.divider()

# --- STOP HERE IF SETUP FAILED ---
if not modules_imported_successfully:
    st.error("### 🚨 Application Setup Failed!")
    st.error(import_error_message)
    st.warning("This usually means one of two things:")
    st.markdown("""
    1.  **Missing Dependencies:** You need to install the required packages.
    2.  **Configuration Error:** The `.env` file is missing or misconfigured.

    **Please follow these steps in your terminal:**
    ```bash
    # 1. Install all required packages
    pip install -r requirements.txt

    # 2. Create your environment file from the example
    cp .env.example .env

    # 3. IMPORTANT: Edit the new .env file with your details
    #    You MUST provide a real email for GEOCODING_USER_AGENT
    #    and any necessary API keys (e.g., GOOGLE_API_KEY).
    ```
    """)
    st.info(
        f"**Error Details:**\n```\n{original_error}\n```\n\n"
        f"**Python Path:** `{sys.path[0]}`\n\n"
        "After fixing the issue, you may need to restart the Streamlit process."
    )
    st.stop()

# --- Initial Data Fetch and API Status Display ---
# Always attempt to update metrics and cache. This will set st.session_state.api_is_reachable.
update_dashboard_metrics_and_cache()

# Display API status based on the result of the first API call
st.sidebar.header("Agent Status")
if st.session_state.api_is_reachable:
    st.sidebar.success(f"Backend API is reachable.")
else:
    st.sidebar.error(f"Backend API is unreachable.")
    st.error(f"**Could not connect to the backend API at `{API_BASE_URL}`.**")
    st.warning("Please ensure the FastAPI backend server is running. You can start it with `./run_api.sh` or by running `run_all.sh` in your terminal.")
    st.info("The UI will automatically try to reconnect. Some features will be disabled until a connection is established.")
    # Clear caches if API is confirmed unreachable
    st.session_state.all_incidents_from_api = []
    st.session_state.filtered_incidents_cache = []


st.sidebar.divider()
st.sidebar.header("Data Ingestion")

# Clear inputs if needed
if st.session_state.get('clear_inputs_on_rerun', False):
    st.session_state.json_input_area_val = ""
    st.session_state.alert_text_input_area_val = ""
    st.session_state.ocr_text_output = ""
    st.session_state.clear_inputs_on_rerun = False

ingest_tab1, ingest_tab2, ingest_tab3 = st.sidebar.tabs(
    ["EIDO JSON", "Raw Text", "Image (OCR)"])
with ingest_tab1:
    uploaded_files = st.file_uploader(
        "Upload EIDO JSON File(s)", type="json", accept_multiple_files=True, key="file_uploader_key",
        disabled=not st.session_state.api_is_reachable)
    json_input_area = st.text_area(
        "Paste EIDO JSON", key="json_input_area_val", height=150,
        disabled=not st.session_state.api_is_reachable)
    sample_dir = os.path.join(PROJECT_ROOT, 'sample_eido')
    selected_sample = st.selectbox("Or Load Sample EIDO:", options=[
                                   "-- Select Sample --"] + list_files_in_dir(sample_dir), key="sample_select_key",
                                   disabled=not st.session_state.api_is_reachable)
with ingest_tab2:
    alert_text_input_area = st.text_area(
        "Paste Raw Alert Text", key="alert_text_input_area_val", height=200,
        disabled=not st.session_state.api_is_reachable)
with ingest_tab3:
    uploaded_image_ocr = st.file_uploader(
        "Upload Image for OCR", type=["png", "jpg", "jpeg"],
        # modules_imported_successfully check is now handled by st.stop()
        disabled=not st.session_state.api_is_reachable)
    if uploaded_image_ocr:
        if st.button("Extract Text from Image", key="ocr_extract_button",
                     # modules_imported_successfully check is now handled by st.stop()
                     disabled=not st.session_state.api_is_reachable):
            with st.spinner("Performing OCR..."):
                # Use extract_text_from_image directly as it's guaranteed to be imported or app stopped
                ocr_text = extract_text_from_image(
                    BytesIO(uploaded_image_ocr.getvalue()))
                if ocr_text:
                    st.session_state.alert_text_input_area_val = ocr_text
                    st.success(
                        "OCR successful! Text placed in 'Raw Alert Text' tab.")
                else:
                    st.error(
                        "OCR failed. Is Tesseract installed and in PATH?")
                get_captured_logs()

if st.sidebar.button("Process Inputs", type="primary", use_container_width=True,
                     disabled=not st.session_state.api_is_reachable):
    status_placeholder = st.sidebar.empty()
    status_placeholder.info("Sending inputs to backend for processing...")

    with st.spinner('Agent is processing...'):
        # Process JSON
        json_to_process = []
        if st.session_state.json_input_area_val:
            try:
                json_to_process.append(json.loads(
                    st.session_state.json_input_area_val))
            except json.JSONDecodeError:
                st.error("Pasted JSON is invalid.")
        if st.session_state.sample_select_key != "-- Select Sample --":
            with open(os.path.join(sample_dir, st.session_state.sample_select_key), 'r') as f:
                json_to_process.append(json.load(f))
        for uf in uploaded_files:
            json_to_process.append(json.loads(uf.getvalue()))

        for item in json_to_process:
            make_api_request("POST", "/api/v1/ingest", payload=item)

        # Process Raw Text
        if st.session_state.alert_text_input_area_val:
            make_api_request("POST", "/api/v1/ingest_alert",
                             payload={"alert_text": st.session_state.alert_text_input_area_val})

    status_placeholder.success("Processing complete!")
    st.session_state.clear_inputs_on_rerun = True
    update_dashboard_metrics_and_cache()
    get_captured_logs()
    time.sleep(1)
    st.rerun()

st.sidebar.divider()
with st.sidebar.expander("Admin Actions"):
    if st.button("Clear All Incidents", use_container_width=True,
                 disabled=not st.session_state.api_is_reachable):
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

# --- Main Dashboard (Continued) ---

metric_cols = st.columns(3)
metric_cols[0].metric("Total Incidents", st.session_state.total_incidents)
metric_cols[1].metric("Active Incidents", st.session_state.active_incidents)
report_counts = [len(inc.reports_core_data)
                 for inc in st.session_state.all_incidents_from_api]
avg_reports = sum(report_counts) / len(report_counts) if report_counts else 0
metric_cols[2].metric("Avg Reports/Incident", f"{avg_reports:.1f}")
st.divider()

st.subheader("Incident Data")

if not st.session_state.all_incidents_from_api:
    if st.session_state.api_is_reachable:
        st.info("No incident data loaded. Please ingest data using the sidebar.")
    else:
        st.warning("No incident data available. Backend API is unreachable.")
    # No st.stop() here, allow the rest of the UI to render with empty data
else:
    # --- Filtering ---
    filter_col1, filter_col2, filter_col3 = st.columns([0.4, 0.3, 0.3])
    all_incidents = st.session_state.all_incidents_from_api
    available_types = sorted(
        list(set(inc.incident_type for inc in all_incidents if inc.incident_type)))
    available_statuses = sorted(
        list(set(inc.status for inc in all_incidents if inc.status)))
    available_zips = sorted(
        list(set(zip_code for inc in all_incidents for zip_code in inc.zip_codes)))

    def update_filters():
        st.session_state.active_filters['types'] = st.session_state.get(
            'filter_type_ms', [])
        st.session_state.active_filters['statuses'] = st.session_state.get(
            'filter_status_ms', [])
        st.session_state.active_filters['zips'] = st.session_state.get(
            'filter_zip_ms', [])

    with filter_col1:
        st.multiselect("Filter by Type:", options=available_types,
                       key="filter_type_ms", on_change=update_filters)
    with filter_col2:
        st.multiselect("Filter by Status:", options=available_statuses,
                       key="filter_status_ms", on_change=update_filters)
    with filter_col3:
        st.multiselect("Filter by ZIP Code:", options=available_zips,
                       key="filter_zip_ms", on_change=update_filters)
    st.divider()

    filtered_incidents = st.session_state.filtered_incidents_cache

    # --- TABS ---
    tab_list, tab_map, tab_details, tab_tools = st.tabs(
        ["Incident Feed", "Incident Map", "Incident Details", "Agentic Tools"])

    with tab_list:
        st.caption(
            f"Displaying {len(filtered_incidents)} incidents based on filters.")
        if filtered_incidents:
            df_list = pd.DataFrame([{
                "ID": inc.incident_id[:8], "Type": inc.incident_type, "Status": inc.status,
                "Last Update": inc.last_updated_at, "Reports": len(inc.reports_core_data),
                "Summary": inc.summary[:100] + '...' if inc.summary else ''
            } for inc in filtered_incidents])
            st.dataframe(df_list, use_container_width=True, hide_index=True, column_config={
                "Last Update": st.column_config.DatetimeColumn(
                    "Last Update",
                    format="YYYY-MM-DD HH:mm:ss",
                )
            })
        else:
            st.info("No incidents match the current filters.")

    with tab_map:
        map_points = []
        for inc in filtered_incidents:
            if inc.locations:
                for lat, lon in inc.locations:
                    map_points.append(
                        {'lat': lat, 'lon': lon, 'tooltip': f"ID: {inc.incident_id[:8]}\nType: {inc.incident_type}"})
        if map_points:
            df_map = pd.DataFrame(map_points)
            view_state = pdk.ViewState(latitude=df_map['lat'].mean(
            ), longitude=df_map['lon'].mean(), zoom=11, pitch=45)
            layer = pdk.Layer('ScatterplotLayer', data=df_map,
                              get_position='[lon, lat]', get_color='[200, 30, 0, 160]', get_radius=100, pickable=True)
            st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v10',
                            initial_view_state=view_state, layers=[layer], tooltip={"text": "{tooltip}"}))
        else:
            st.info("No geocoded locations to display for the current filter.")

    with tab_details:
        if filtered_incidents:
            options = {
                f"{inc.incident_id[:8]} - {inc.incident_type}": inc for inc in filtered_incidents}
            selected_key = st.selectbox(
                "Select Incident:", options=list(options.keys()))
            if selected_key:
                selected_incident = options[selected_key]
                st.subheader(f"Incident: {selected_incident.incident_id}")
                st.json(selected_incident.model_dump_json(indent=2))
        else:
            st.info("No incidents to display details for based on current filters.")

    with tab_tools:
        st.subheader("Agentic Tools")
        tool_eido, tool_geocode = st.tabs(
            ["EIDO Generator", "Local Geocoding Store"])

        with tool_eido:
            st.write("#### EIDO Generator")
            st.caption(
                "Create compliant EIDO JSON examples from scenario descriptions.")
            template_dir = os.path.join(PROJECT_ROOT, 'eido_templates')
            templates = list_files_in_dir(template_dir)
            selected_template = st.selectbox("Select Template", options=templates,
                                             disabled=not st.session_state.api_is_reachable)
            scenario = st.text_area("Scenario Description",
                                    disabled=not st.session_state.api_is_reachable)
            if st.button("Generate EIDO", disabled=not st.session_state.api_is_reachable):
                if selected_template and scenario:
                    payload = {"template_name": selected_template,
                               "scenario_description": scenario}
                    response = make_api_request(
                        "POST", "/api/v1/generate_eido_from_template", payload=payload)
                    if response and 'generated_eido' in response:
                        st.session_state.generated_eido_json = json.dumps(
                            response['generated_eido'], indent=2)
            if st.session_state.get('generated_eido_json'):
                st_ace(value=st.session_state.generated_eido_json,
                       language='json', readonly=True, height=300)
            else:
                if st.session_state.api_is_reachable:
                    st.info(
                        "Enter a scenario and select a template to generate EIDO JSON.")
                else:
                    st.warning(
                        "EIDO Generator is disabled because the backend API is unreachable.")

        with tool_geocode:
            st.write("#### Manage Local Geocoding Store")
            st.caption(
                "Add or remove custom location names for the advanced geocoding service.")

            def fetch_local_locations():
                data = make_api_request(
                    "GET", "/api/v1/tools/geocoding/local_store")
                if data and isinstance(data, dict):
                    st.session_state.local_geocoded_locations = data
                else:
                    st.session_state.local_geocoded_locations = {}

            if st.button("Refresh Locations", key="refresh_geolocations", disabled=not st.session_state.api_is_reachable):
                fetch_local_locations()

            locations = st.session_state.get('local_geocoded_locations', {})
            if not locations and st.session_state.api_is_reachable:
                fetch_local_locations()
                locations = st.session_state.get(
                    'local_geocoded_locations', {})

            st.markdown("##### Add or Update a Location")
            with st.form("add_location_form", clear_on_submit=True):
                add_name = st.text_input(
                    "Location Name (e.g., Geisel Library, Warren Mall)")
                c1, c2 = st.columns(2)
                add_lat = c1.number_input(
                    "Latitude", format="%.6f", value=0.0)
                add_lon = c2.number_input(
                    "Longitude", format="%.6f", value=0.0)
                add_notes = st.text_input("Notes (optional)")
                submitted = st.form_submit_button(
                    "Add/Update Location", disabled=not st.session_state.api_is_reachable)

                if submitted:
                    if add_name and add_lat is not None and add_lon is not None:
                        payload = {"location_name": add_name, "latitude": add_lat,
                                   "longitude": add_lon, "notes": add_notes}
                        if make_api_request("POST", "/api/v1/tools/geocoding/local_store", payload=payload):
                            st.success(f"Location '{add_name}' added/updated.")
                            fetch_local_locations()
                            st.rerun()
                    else:
                        st.warning(
                            "Please fill in Name, Latitude, and Longitude.")

            st.markdown("##### Existing Locations")
            if locations:
                loc_list = [{"Location Name": name, **details}
                            for name, details in locations.items()]
                st.dataframe(pd.DataFrame(loc_list), use_container_width=True, hide_index=True, column_config={
                    "lat": "Latitude", "lon": "Longitude"
                })

                st.markdown("---")
                st.write("###### Remove a Location")
                loc_to_delete = st.selectbox(
                    "Select location to remove", options=[""] + sorted(list(locations.keys())))
                if loc_to_delete:
                    if st.button(f"Delete '{loc_to_delete}'", type="secondary", disabled=not st.session_state.api_is_reachable):
                        encoded_loc_name = urllib.parse.quote(loc_to_delete)
                        if make_api_request("DELETE", f"/api/v1/tools/geocoding/local_store/{encoded_loc_name}"):
                            st.success(f"Location '{loc_to_delete}' removed.")
                            fetch_local_locations()
                            st.rerun()
            else:
                st.info("No custom locations in the local store.")

st.divider()
st.caption(f"EIDO Sentinel v0.9.1 | End of Dashboard")
