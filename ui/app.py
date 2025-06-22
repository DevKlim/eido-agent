import streamlit as st
import json
import os
import pandas as pd
import time
from datetime import datetime, timezone
import sys
import logging
from io import StringIO
import altair as alt
import pydeck as pdk
from typing import List, Dict, Optional, Any, Set
import requests
from PIL import Image
from pydantic import ValidationError
import urllib.parse

# --- Page Configuration ---
try:
    PAGE_ICON_PATH = os.path.abspath(os.path.join(os.path.dirname(
        __file__), '..', 'static', 'images', 'logo_icon_light.png'))
    page_icon_img = Image.open(PAGE_ICON_PATH)
except FileNotFoundError:
    page_icon_img = "🤖"

st.set_page_config(
    layout="wide",
    page_title="EIDO Sentinel | Agentic Incident Processor",
    page_icon=page_icon_img,
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': "https://github.com/DevKlim/eido-sentinel",
        'Report a bug': "https://github.com/DevKlim/eido-sentinel/issues",
        'About': "# EIDO Sentinel v1.0.0\nAgent Driven Emergency Incident Processor. A link to the project showcase is in the sidebar."
    }
)

# --- Defensive Setup with Graceful Error Handling ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

modules_imported_successfully = False
import_error_message = ""
original_error = None
local_settings = None

log_stream = StringIO()
log_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
stream_handler_ui = logging.StreamHandler(log_stream)
stream_handler_ui.setFormatter(log_formatter)
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == log_stream for h in root_logger.handlers):
    root_logger.addHandler(stream_handler_ui)

log_level_to_set = 'INFO'
try:
    from config.settings import settings as temp_settings
    local_settings = temp_settings
    log_level_to_set = getattr(local_settings, 'log_level', 'INFO').upper()
    root_logger.setLevel(log_level_to_set)
    from data_models.schemas import Incident as PydanticIncident
    modules_imported_successfully = True
except Exception as e:
    import_error_message = f"A required module failed to import. This is often caused by missing dependencies or a misconfigured '.env' file that prevents 'config/settings.py' from loading."
    original_error = e
    root_logger.setLevel(logging.ERROR)

logger_ui = logging.getLogger("EidoSentinelUI")
if modules_imported_successfully:
    logger_ui.debug("UI Log Capture StreamHandler added to root logger.")

# --- Environment and API Configuration ---
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
if local_settings and "API_BASE_URL" not in os.environ:
    API_BASE_URL = local_settings.api_base_url

logger_ui.info(f"UI configured to use API at: {API_BASE_URL}")

LANDING_PAGE_URL = API_BASE_URL.rsplit(':', 1)[0] if ':' in API_BASE_URL.rsplit('/',1)[-1] else API_BASE_URL

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'log_messages': [],
        'total_incidents': 0, 'active_incidents': 0,
        'clear_inputs_on_rerun': False,
        'filtered_incidents_cache': [], 'active_filters': {},
        'all_incidents_from_api': [],
        'api_is_reachable': None,
        'json_input_area_val': "", 'alert_text_input_area_val': "",
        'active_view': 'Incident Feed', 'selected_incident_id': None,
        'force_data_refresh': True,
        'eido_schema_cache': None,
        'template_builder_selections': set(),
        'generated_template_cache': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.api_base_url = API_BASE_URL

init_session_state()

# --- API Helper Functions ---
def make_api_request(method: str, endpoint: str, payload: Optional[Dict] = None, params: Optional[Dict] = None, is_critical: bool = False) -> Optional[Any]:
    url = f"{st.session_state.api_base_url}{endpoint}"
    try:
        response = requests.request(method.upper(), url, json=payload, params=params, timeout=30)
        st.session_state.api_is_reachable = True 
        response.raise_for_status()
        if response.status_code == 204:
            return True
        return response.json() if response.content else True
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error ({e.response.status_code}): {e.response.text}")
        logger_ui.error(f"API HTTP Error for {url}: {e.response.status_code} - {e.response.text}", exc_info=False)
    except requests.exceptions.RequestException as e:
        logger_ui.critical(f"API Connection Error for {url}: {e}", exc_info=False)
        st.session_state.api_is_reachable = False
        if is_critical:
            st.error(f"Critical API connection failed: {e}. The app may not function correctly.")
    return None

# --- UI Helper & Data Functions ---
def get_captured_logs():
    log_stream.seek(0)
    logs_captured = log_stream.read()
    log_stream.truncate(0)
    log_stream.seek(0)
    if new_entries := [entry for entry in logs_captured.strip().split('\n') if entry.strip()]:
        st.session_state.log_messages = new_entries + st.session_state.log_messages[:199]

def fetch_and_cache_all_incidents():
    if not modules_imported_successfully:
        st.session_state.all_incidents_from_api = []
        return
    
    data = make_api_request("GET", "/api/v1/incidents", is_critical=True)
    
    if data and isinstance(data, list):
        incidents = []
        for i, inc_data in enumerate(data):
            try:
                incidents.append(PydanticIncident(**inc_data))
            except ValidationError as e:
                st.error(f"Data parsing error for incident #{i+1}. The data from the API does not match the expected format.")
                logger_ui.error(f"Pydantic validation error on incident data: {e.errors()}", exc_info=False)
                st.code(json.dumps(inc_data, indent=2), language="json")
                continue
        st.session_state.all_incidents_from_api = sorted(incidents, key=lambda x: x.last_updated_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    elif data is None:
        st.session_state.all_incidents_from_api = []
    
    update_metrics_and_filtered_cache()
    st.session_state.force_data_refresh = False

def update_metrics_and_filtered_cache():
    all_incidents = st.session_state.get('all_incidents_from_api', [])
    st.session_state.total_incidents = len(all_incidents)
    active_statuses = ["active", "updated", "monitoring", "dispatched", "acknowledged", "enroute", "onscene"]
    st.session_state.active_incidents = sum(1 for inc in all_incidents if inc.status and inc.status.lower() in active_statuses)

    filters = st.session_state.get('active_filters', {})
    filtered = all_incidents
    if filters.get('types'):
        filtered = [inc for inc in filtered if inc.incident_type in filters['types']]
    if filters.get('statuses'):
        filtered = [inc for inc in filtered if inc.status in filters['statuses']]
    if filters.get('zips'):
        filtered = [inc for inc in filtered if any(zip_code in filters['zips'] for zip_code in inc.zip_codes)]

    st.session_state.filtered_incidents_cache = filtered

def list_files_in_dir(dir_path, extension=".json"):
    return sorted([f for f in os.listdir(dir_path) if f.endswith(extension)]) if os.path.exists(dir_path) else []

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

# --- Main App Title ---
st.title("EIDO Sentinel Dashboard")
st.caption(f"v1.0.0 | Connected to: {st.session_state.api_base_url}")

# --- STOP HERE IF SETUP FAILED ---
if not modules_imported_successfully:
    st.error("### Application Setup Failed!")
    st.error(import_error_message)
    st.warning("This usually means missing dependencies or a misconfigured `.env` file.")
    st.code(f"Error Details: {original_error}", language='bash')
    st.stop()

# --- DATA FETCHING LOGIC (PERFORMANCE FIX) ---
if st.session_state.get('force_data_refresh', True):
    fetch_and_cache_all_incidents()

st.sidebar.header("Agent Status")
if st.session_state.api_is_reachable:
    st.sidebar.success("Backend API is reachable.")
elif st.session_state.api_is_reachable is False:
    st.sidebar.error("Backend API is unreachable.")

st.sidebar.divider()
st.sidebar.header("Data Ingestion")

if st.session_state.get('clear_inputs_on_rerun', False):
    st.session_state.update(json_input_area_val="", alert_text_input_area_val="")
    st.session_state.clear_inputs_on_rerun = False

ingest_tab1, ingest_tab2 = st.sidebar.tabs(["EIDO JSON", "Raw Text"])
with ingest_tab1:
    uploaded_files = st.file_uploader("Upload EIDO JSON File(s)", type="json", accept_multiple_files=True, disabled=not st.session_state.api_is_reachable)
    json_input_area = st.text_area("Paste EIDO JSON", key="json_input_area_val", height=150, disabled=not st.session_state.api_is_reachable)
    sample_dir = os.path.join(PROJECT_ROOT, 'sample_eido')
    selected_sample = st.selectbox("Or Load Sample EIDO:", options=["-- Select Sample --"] + list_files_in_dir(sample_dir), disabled=not st.session_state.api_is_reachable)

with ingest_tab2:
    alert_text_input_area = st.text_area("Paste Raw Alert Text", key="alert_text_input_area_val", height=200, disabled=not st.session_state.api_is_reachable)

if st.sidebar.button("Process Inputs", type="primary", use_container_width=True, disabled=not st.session_state.api_is_reachable):
    with st.spinner('Agent is processing... This may take a moment.'):
        processing_error = False
        json_to_process = []
        if st.session_state.json_input_area_val:
            try:
                json_to_process.append(json.loads(st.session_state.json_input_area_val))
            except json.JSONDecodeError:
                st.error("Pasted JSON is invalid.")
                processing_error = True
        if selected_sample != "-- Select Sample --":
            with open(os.path.join(sample_dir, selected_sample), 'r') as f:
                json_to_process.append(json.load(f))
        for uf in uploaded_files:
            json_to_process.append(json.loads(uf.getvalue()))
        
        if not processing_error:
            for item in json_to_process:
                if make_api_request("POST", "/api/v1/ingest", payload=item) is None:
                    processing_error = True
                    break
        
        if not processing_error and st.session_state.alert_text_input_area_val:
            if make_api_request("POST", "/api/v1/ingest_alert", payload={"alert_text": st.session_state.alert_text_input_area_val}) is None:
                processing_error = True

    if not processing_error:
        st.sidebar.success("Processing complete!")
        st.session_state.clear_inputs_on_rerun = True
        st.session_state.force_data_refresh = True
        st.rerun()

st.sidebar.divider()
with st.sidebar.expander("Admin Actions"):
    if st.button("Clear All Incidents", use_container_width=True, disabled=not st.session_state.api_is_reachable):
        if make_api_request("DELETE", "/api/v1/admin/clear_store"):
            st.success("Incident store cleared.")
            st.session_state.force_data_refresh = True
            st.rerun()

with st.sidebar.expander("Processing Log"):
    get_captured_logs()
    st.code("\n".join(st.session_state.log_messages), language='log')

st.divider()

# --- METRICS and FILTERS ---
metric_cols = st.columns(3)
metric_cols[0].metric("Total Incidents", st.session_state.total_incidents)
metric_cols[1].metric("Active Incidents", st.session_state.active_incidents)
report_counts = [len(inc.reports_core_data) for inc in st.session_state.all_incidents_from_api]
metric_cols[2].metric("Avg Reports/Incident", f"{sum(report_counts) / len(report_counts) if report_counts else 0:.1f}")
st.divider()

if st.session_state.all_incidents_from_api:
    st.markdown("##### Filter Controls")
    filter_col1, filter_col2, filter_col3 = st.columns([0.4, 0.3, 0.3])
    all_incidents = st.session_state.all_incidents_from_api
    available_types = sorted(list(set(inc.incident_type for inc in all_incidents if inc.incident_type)))
    available_statuses = sorted(list(set(inc.status for inc in all_incidents if inc.status)))
    available_zips = sorted(list(set(zip_code for inc in all_incidents for zip_code in inc.zip_codes)))

    def update_filters():
        st.session_state.active_filters['types'] = st.session_state.filter_type_ms
        st.session_state.active_filters['statuses'] = st.session_state.filter_status_ms
        st.session_state.active_filters['zips'] = st.session_state.filter_zip_ms
        update_metrics_and_filtered_cache()

    filter_col1.multiselect("Filter by Type:", options=available_types, key="filter_type_ms", on_change=update_filters, default=st.session_state.get('active_filters', {}).get('types'))
    filter_col2.multiselect("Filter by Status:", options=available_statuses, key="filter_status_ms", on_change=update_filters, default=st.session_state.get('active_filters', {}).get('statuses'))
    filter_col3.multiselect("Filter by ZIP Code:", options=available_zips, key="filter_zip_ms", on_change=update_filters, default=st.session_state.get('active_filters', {}).get('zips'))
st.divider()

# --- MAIN VIEW RENDER FUNCTIONS ---
def render_dashboard():
    st.subheader("Analytics Dashboard")
    incidents = st.session_state.filtered_incidents_cache
    if not incidents:
        st.info("No incident data to display. Please adjust filters or ingest data.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Incidents by ZIP Code")
        zip_codes = [zip_code for inc in incidents for zip_code in inc.zip_codes if zip_code]
        if zip_codes:
            zip_df = pd.DataFrame(zip_codes, columns=['zip_code']).value_counts().reset_index(name='count').rename(columns={'zip_code': 'ZIP Code', 'count': 'Incidents'})
            chart = alt.Chart(zip_df).mark_bar().encode(x=alt.X('ZIP Code:N', sort='-y'), y='Incidents:Q', tooltip=['ZIP Code', 'Incidents']).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No ZIP code data available.")
    with col2:
        st.markdown("##### Incidents by Type")
        types = [inc.incident_type for inc in incidents if inc.incident_type]
        if types:
            type_df = pd.DataFrame(types, columns=['type']).value_counts().reset_index(name='count').rename(columns={'type': 'Type', 'count': 'Incidents'})
            chart = alt.Chart(type_df).mark_bar().encode(x=alt.X('Type:N', sort='-y'), y='Incidents:Q', color=alt.Color('Type:N').legend(None), tooltip=['Type', 'Incidents']).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No incident type data available.")

def render_incident_feed():
    st.subheader("Incident Feed")
    st.caption(f"Displaying {len(st.session_state.filtered_incidents_cache)} incidents. Click an incident to inspect.")
    if not st.session_state.filtered_incidents_cache:
        st.info("No incidents match the current filters.")
        return

    for inc in st.session_state.filtered_incidents_cache:
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 3, 2])
            c1.markdown(f"**{inc.name or 'Untitled Incident'}**")
            c1.caption(f"Type: `{inc.incident_type or 'N/A'}` | ID: `{inc.incident_id[:8]}`")
            c2.markdown(f"**Status:** `{inc.status}`")
            c2.caption(f"Last Update: `{inc.last_updated_at.strftime('%Y-%m-%d %H:%M') if inc.last_updated_at else 'N/A'}`")
            if c3.button("View Details", key=f"btn_{inc.incident_id}", use_container_width=True):
                st.session_state.selected_incident_id = inc.incident_id
                st.rerun()

def render_incident_details():
    if st.button(f"⬅️ Back to {st.session_state.get('active_view', 'Incident Feed')}"):
        st.session_state.selected_incident_id = None
        st.rerun()
    
    st.subheader("Incident Details")
    if not st.session_state.selected_incident_id:
        st.info("Select an incident from the 'Incident Feed' to see details.")
        return

    incident = next((inc for inc in st.session_state.all_incidents_from_api if inc.incident_id == st.session_state.selected_incident_id), None)
    if not incident:
        st.error(f"Could not find incident ID: {st.session_state.selected_incident_id}")
        st.session_state.selected_incident_id = None
        return

    view_tab, export_tab = st.tabs(["Formatted View", "Export Source EIDO"])
    with view_tab:
        st.markdown(f"#### {incident.name or 'Untitled Incident'}")
        st.caption(f"`{incident.incident_id}`")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", incident.status)
        c2.metric("Incident Type", incident.incident_type or "N/A")
        c3.metric("# of Reports", len(incident.reports_core_data))
        st.divider()
        st.markdown("##### **Summary**")
        st.info(incident.summary or "_No summary generated._")
        st.markdown("##### **Next Actions**")
        st.markdown("\n".join(f"- {action}" for action in incident.recommended_actions) or "_No actions recommended._")
        st.divider()
        with st.expander("**Location Information**", expanded=True):
            if incident.locations:
                location_df = pd.DataFrame(incident.locations, columns=["latitude", "longitude"])
                
                view_state = pdk.ViewState(
                    latitude=location_df["latitude"].mean(),
                    longitude=location_df["longitude"].mean(),
                    zoom=14,
                    pitch=50,
                )

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=location_df,
                    get_position=["longitude", "latitude"],
                    get_color="[200, 30, 0, 160]",
                    get_radius=50,
                    pickable=True,
                )

                st.pydeck_chart(pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"html": "<b>Location:</b><br/>Lat: {latitude}<br/>Lon: {longitude}"}
                ))

            st.markdown(f"**Addresses:** `{' | '.join(incident.addresses) or 'N/A'}`")
            st.markdown(f"**ZIP Codes:** `{', '.join(incident.zip_codes) or 'N/A'}`")
        with st.expander("**Full Description History**"):
            st.text_area("History", value=incident.get_full_description_history(), height=250, disabled=True, label_visibility="collapsed")
    
    with export_tab:
        st.markdown("#### Export Source Report as EIDO JSON")
        st.caption("An incident is composed of one or more reports. Select a source report below to view, copy, or download its original EIDO JSON data.")
        
        reports = sorted(incident.reports_core_data, key=lambda r: r.timestamp, reverse=True)
        
        if not any(r.original_eido_dict for r in reports):
            st.warning("This incident has no associated reports with original EIDO data to export.")
        else:
            report_options = {
                f"Report from {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({r.source or 'N/A'})": r 
                for r in reports if r.original_eido_dict
            }
            
            if not report_options:
                 st.warning("This incident has no associated reports with original EIDO data to export.")
                 return

            selected_key = st.selectbox("Select a source report to export:", options=list(report_options.keys()))
            selected_report = report_options.get(selected_key)
            
            if selected_report and selected_report.original_eido_dict:
                json_string = json.dumps(selected_report.original_eido_dict, indent=2)
                st.code(json_string, language="json", line_numbers=True)
                
                st.download_button(
                    label="Download EIDO JSON",
                    data=json_string,
                    file_name=f"EIDO_Report_{selected_report.report_id[:8]}.json",
                    mime="application/json",
                    key=f"download_btn_{selected_report.report_id}"
                )
            elif selected_key:
                st.info("The selected report does not have an original EIDO document attached.")

def render_map_view():
    st.subheader("Incident Map View")
    incidents = st.session_state.filtered_incidents_cache
    
    map_data = []
    for inc in incidents:
        if inc.locations:
            lat, lon = inc.locations[0]
            map_data.append({
                "latitude": lat,
                "longitude": lon,
                "tooltip": f"{inc.name}\nType: {inc.incident_type}\nStatus: {inc.status}"
            })

    if not map_data:
        st.info("No incidents with location data to display on map.")
        return

    df = pd.DataFrame(map_data)
    
    view_state = pdk.ViewState(
        latitude=df["latitude"].mean(),
        longitude=df["longitude"].mean(),
        zoom=11,
        pitch=50,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_color="[200, 30, 0, 160]",
        get_radius=100,
        pickable=True,
    )

    st.pydeck_chart(pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "{tooltip}"}
    ))

def render_geocoding_editor():
    st.subheader("Local Geocoding Store Editor")
    st.info("""
    **How to use this tool:**
    1.  To add a new location, fill the 'Add New Location' form and submit. This is useful for landmarks or places not in public map databases.
    2.  To find and edit an existing location, scroll through the list below or use your browser's find function (Ctrl+F or Cmd+F).
    3.  Changes are saved to `data/geocoded_locations.json` and are immediately available to the agent's geocoding service.
    """)
    st.caption("Manage custom location-to-coordinate mappings. These entries are prioritized by the geocoding service.")

    locations_data = make_api_request("GET", "/api/v1/tools/geocoding/local_store")

    if locations_data is None:
        st.error("Could not fetch locations from the backend. The service may be down.")
        return

    editor_tabs = st.tabs(["Manage Entries", "View Raw JSON"])

    with editor_tabs[0]:
        with st.expander("➕ Add New Location", expanded=False):
            with st.form("new_location_form"):
                new_loc_name = st.text_input("Location Name (e.g., 'Geisel Library Entrance')")
                col1, col2 = st.columns(2)
                new_lat = col1.number_input("Latitude", format="%.6f")
                new_lon = col2.number_input("Longitude", format="%.6f")
                new_source = st.text_input("Source", value="manual_ui_input")
                new_notes = st.text_area("Notes")
                
                submitted = st.form_submit_button("Add Location")
                if submitted:
                    if not new_loc_name or new_lat == 0.0 or new_lon == 0.0:
                        st.warning("Please fill in Location Name, Latitude, and Longitude.")
                    else:
                        payload = {"location_name": new_loc_name, "latitude": new_lat, "longitude": new_lon, "source": new_source, "notes": new_notes}
                        if make_api_request("POST", "/api/v1/tools/geocoding/local_store", payload=payload):
                            st.success(f"Location '{new_loc_name}' added successfully.")
                            st.rerun()
        st.divider()
        st.markdown(f"**{len(locations_data)} Existing Locations**")
        if not locations_data:
            st.info("No custom locations in the store. Add one using the form above.")
        else:
            sorted_locations = sorted(locations_data.items())
            for name, data in sorted_locations:
                with st.container(border=True):
                    col_disp1, col_disp2 = st.columns([3, 1])
                    with col_disp1:
                        st.markdown(f"**{name}**")
                        st.caption(f"Source: `{data.get('source', 'N/A')}` | Updated: `{data.get('last_updated', 'N/A').split('T')[0] if data.get('last_updated') else 'N/A'}`")
                        st.code(f"Lat: {data.get('lat', 0.0):.6f}, Lon: {data.get('lon', 0.0):.6f}", language='bash')
                    
                    with col_disp2:
                        if st.button("Delete", key=f"delete_{name}", use_container_width=True, type="secondary"):
                            encoded_name = urllib.parse.quote(name)
                            if make_api_request("DELETE", f"/api/v1/tools/geocoding/local_store/{encoded_name}"):
                                st.success(f"Location '{name}' deleted.")
                                st.rerun()

                    with st.expander("Edit"):
                        with st.form(key=f"form_edit_{name}"):
                            edit_col1, edit_col2 = st.columns(2)
                            edited_lat = edit_col1.number_input("Latitude", value=data.get('lat', 0.0), format="%.6f", key=f"lat_{name}")
                            edited_lon = edit_col2.number_input("Longitude", value=data.get('lon', 0.0), format="%.6f", key=f"lon_{name}")
                            edited_source = st.text_input("Source", value=data.get('source', ''), key=f"source_{name}")
                            edited_notes = st.text_area("Notes", value=data.get('notes', ''), key=f"notes_{name}")
                            
                            if st.form_submit_button("Update Location", use_container_width=True):
                                payload = {"location_name": name, "latitude": edited_lat, "longitude": edited_lon, "source": edited_source, "notes": edited_notes}
                                if make_api_request("POST", "/api/v1/tools/geocoding/local_store", payload=payload):
                                    st.success(f"Location '{name}' updated.")
                                    st.rerun()
    
    with editor_tabs[1]:
        st.markdown("#### Raw `geocoded_locations.json` Content")
        st.caption("This is the direct content of the JSON file used by the local geocoder.")
        if locations_data:
            st.code(json.dumps(locations_data, indent=2), language='json')
        else:
            st.info("The geocoding store is empty.")

def render_eido_template_editor():
    """Renders the UI for creating new EIDO templates with live preview."""
    st.subheader("EIDO Template Editor")
    st.info("""
    **How to use this tool:**
    1.  Give your template a unique filename ending in `.json`.
    2.  Use the search box to find and select the EIDO components you need for your incident type (e.g., 'vehicleComponent' for a traffic collision).
    3.  As you select components, the JSON preview on the right updates automatically.
    4.  Once you are satisfied with the structure, click 'Save Template to Server'.
    5.  The new template will then be available for the agent's raw text parsing pipeline.
    """)

    # Fetch schema on first load
    if st.session_state.eido_schema_cache is None:
        with st.spinner("Loading EIDO schema from backend..."):
            schema_data = make_api_request("GET", "/api/v1/tools/eido/schema")
            if schema_data:
                st.session_state.eido_schema_cache = schema_data
            else:
                st.error("Failed to load EIDO schema. The editor cannot be displayed.")
                return

    schema = st.session_state.eido_schema_cache
    root_component_name = "EmergencyIncidentDataObjectType"
    root_def = schema.get(root_component_name)

    if not root_def:
        st.error(f"Could not find root component '{root_component_name}' in schema.")
        return

    # Helper to recursively create a placeholder dictionary for a component
    @st.cache_data(show_spinner=False)
    def create_placeholder_object(_comp_name: str, _all_schemas: Dict) -> Dict:
        comp_def = _all_schemas.get(_comp_name, {})
        placeholder = {}
        properties_to_process = {}

        if 'allOf' in comp_def:
            for part in comp_def['allOf']:
                if '$ref' in part:
                    ref_name = part['$ref'].split('/')[-1]
                    parent_placeholder = create_placeholder_object(ref_name, _all_schemas)
                    placeholder.update(parent_placeholder)
                elif 'properties' in part:
                    properties_to_process.update(part.get('properties', {}))
        
        properties_to_process.update(comp_def.get('properties', {}))
        
        for prop_name, prop_schema in properties_to_process.items():
            if '$ref' in prop_schema:
                 ref_name = prop_schema["$ref"].split("/")[-1]
                 placeholder[prop_name] = create_placeholder_object(ref_name, _all_schemas)
            elif prop_schema.get("type") == "array" and "$ref" in prop_schema.get("items", {}):
                 ref_name = prop_schema.get("items", {})["$ref"].split("/")[-1]
                 placeholder[prop_name] = [create_placeholder_object(ref_name, _all_schemas)]
            else:
                placeholder[prop_name] = f"[{prop_name.upper()}]"
        return placeholder

    # UI layout
    editor_col, preview_col = st.columns(2)

    with editor_col:
        st.markdown("##### 1. Configure Template")
        filename = st.text_input("Template Filename", "my_new_template.json", help="Must end with .json")

        st.markdown("##### 2. Select Components")
        st.caption(f"The root **{root_component_name}** is always included.")

        root_properties = {}
        if 'allOf' in root_def:
            for part in root_def.get('allOf', []):
                if '$ref' in part:
                    ref_name = part['$ref'].split('/')[-1]
                    parent_def = schema.get(ref_name, {})
                    if 'properties' in parent_def:
                        root_properties.update(parent_def['properties'])
                if 'properties' in part:
                    root_properties.update(part['properties'])
        root_properties.update(root_def.get('properties', {}))
        
        available_components = {}
        for prop_name, prop_schema in root_properties.items():
            is_array = prop_schema.get("type") == "array"
            ref_path = prop_schema.get("items", {}).get("$ref") if is_array else prop_schema.get("$ref")
            if ref_path:
                ref_name = ref_path.split("/")[-1]
                available_components[prop_name] = {'ref': ref_name, 'is_array': is_array}
        
        search_term = st.text_input("Search Components", help="Filter the list by component name (e.g., 'vehicle')")
        filtered_component_names = [
            name for name in sorted(available_components.keys()) 
            if search_term.lower() in name.lower()
        ]

        st.multiselect(
            "Select optional top-level components:",
            options=filtered_component_names,
            key="template_builder_selections",
            format_func=lambda x: f"{x} ({available_components[x]['ref']})",
            help="Choose the main data blocks for your template."
        )

    # Live template generation logic
    base_template = {}
    required_fields = root_def.get('required', [])
    for key, value in root_properties.items():
        if key in required_fields or key not in available_components:
            if key not in base_template:
                 base_template[key] = create_placeholder_object(key, schema) if '$ref' in value else f"[{key.upper()}]"

    for prop_name in st.session_state.template_builder_selections:
        if prop_name in root_properties:
            prop_schema = root_properties[prop_name]
            is_array = prop_schema.get("type") == "array"
            ref_path = prop_schema.get("items", {}).get("$ref") if is_array else prop_schema.get("$ref")
            if ref_path:
                ref_name = ref_path.split("/")[-1]
                placeholder = create_placeholder_object(ref_name, schema)
                base_template[prop_name] = [placeholder] if is_array else placeholder

    st.session_state.generated_template_cache = base_template

    with preview_col:
        st.markdown("##### 3. Live Preview and Save")
        if st.session_state.generated_template_cache:
            try:
                template_json_str = json.dumps(st.session_state.generated_template_cache, indent=2)
                st.code(template_json_str, language="json", height=500)
            
                if st.button("Save Template to Server", type="primary"):
                    payload = {"filename": filename, "content": st.session_state.generated_template_cache}
                    response = make_api_request("POST", "/api/v1/tools/eido/templates", payload=payload)
                    if response:
                        st.success(f"Template '{filename}' saved successfully on the server!")
            except TypeError as e:
                st.error(f"Error generating JSON preview: {e}")
                st.code(str(st.session_state.generated_template_cache))
        else:
            st.info("Select components on the left to see a live preview here.")


# --- MAIN NAVIGATION AND VIEW RENDERING ---
last_active_view = st.session_state.get('active_view', 'Incident Feed')
view_options = ["Incident Feed", "Dashboard", "Map View", "Geocoding Editor", "Template Editor"]
if last_active_view not in view_options: last_active_view = 'Incident Feed'
view_index = view_options.index(last_active_view)

selected_view = st.radio(
    "Navigation", options=view_options, key="navigation_radio", 
    horizontal=True, label_visibility="collapsed", index=view_index
)

if selected_view != st.session_state.active_view:
    st.session_state.selected_incident_id = None
    st.session_state.active_view = selected_view
    st.rerun()

if st.session_state.get('selected_incident_id'):
    render_incident_details()
else:
    view_render_map = {
        "Incident Feed": render_incident_feed,
        "Dashboard": render_dashboard,
        "Map View": render_map_view,
        "Geocoding Editor": render_geocoding_editor,
        "Template Editor": render_eido_template_editor,
    }
    render_func = view_render_map.get(st.session_state.active_view, render_incident_feed)
    render_func()

st.divider()
st.caption("EIDO Sentinel v1.0.0 | End of Dashboard")