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
from typing import List, Dict, Optional # Added Optional

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="EIDO Sentinel | AI Incident Processor",
    page_icon="🚨",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/LXString/eido-sentinel', # Replace with your repo URL
        'Report a bug': "https://github.com/LXString/eido-sentinel/issues", # Replace with your repo URL
        'About': "# EIDO Sentinel\nAI-Powered Emergency Incident Processor POC."
    }
)

# --- Setup Python Path & Imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
modules_imported_successfully = True
import_error_message = ""
original_error = None
try:
    from config.settings import settings
    logging.basicConfig(level=settings.log_level.upper(), format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logger_ui = logging.getLogger(__name__) # Define logger_ui here
    from agent.agent_core import eido_agent_instance
    from services.storage import incident_store
    from data_models.schemas import Incident
    # Import the new LLM function
    from agent.llm_interface import fill_eido_template
except Exception as e:
    modules_imported_successfully = False
    import_error_message = f"Setup Error: {e}"
    original_error = e
    # Use print if logger failed to initialize
    print(f"CRITICAL SETUP ERROR: {import_error_message}")
    if original_error: print(original_error)


if not modules_imported_successfully:
    st.error(f"🚨 **CRITICAL ERROR:** Failed during application setup.")
    st.error(f"**Details:** {import_error_message}")
    if original_error: st.exception(original_error)
    st.warning("Please ensure dependencies are installed (`pip install -r requirements.txt`) and the project structure is correct.")
    st.info("Run the app from the project's root directory: `streamlit run ui/app.py`")
    st.stop()

# --- Log Capture Setup ---
log_stream = StringIO()
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setFormatter(log_formatter)
root_logger = logging.getLogger()
root_logger.setLevel(settings.log_level.upper())
if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == log_stream for h in root_logger.handlers):
    root_logger.addHandler(stream_handler)

# --- Global Variables & Constants ---
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sample_eido'))
TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eido_templates')) # Define template directory

# Create directories if they don't exist
for dir_path in [SAMPLE_DIR, TEMPLATE_DIR]:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger_ui.info(f"Created directory: {dir_path}")
        except Exception as e:
            st.error(f"Failed to create directory {dir_path}: {e}")
            logger_ui.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)


# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'incidents_df': pd.DataFrame(columns=["ID", "Type", "Status", "Reports", "Last Update", "Summary", "Actions", "Locations"]),
        'log_messages': [],
        'map_data': pd.DataFrame(columns=['lat', 'lon', 'incident_id', 'type']),
        'total_incidents': 0,
        'active_incidents': 0,
        'settings_saved': False,
        'total_reports_geo_checked': 0,
        'reports_with_geo': 0,
        'clear_inputs_on_rerun': False,
        'generated_eido_json': None, # State for EIDO generator output
        'filtered_incidents': [], # Store filtered incidents for reuse across tabs
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Load initial settings
    settings_keys_to_sync = [
        'llm_provider', 'google_api_key', 'google_model_name',
        'openrouter_api_key', 'openrouter_model_name', 'openrouter_api_base_url',
        'local_llm_api_key', 'local_llm_model_name', 'local_llm_api_base_url',
        'geocoding_user_agent',
    ]
    for key in settings_keys_to_sync:
        if key not in st.session_state:
            st.session_state[key] = getattr(settings, key, None)
    if 'google_model_options' not in st.session_state:
        st.session_state.google_model_options = settings.google_model_options

init_session_state()

# --- Helper Functions ---
def list_files_in_dir(dir_path, extension=".json"):
    """Lists files with a specific extension in a directory."""
    if os.path.exists(dir_path):
        try:
            files = [f for f in os.listdir(dir_path) if f.endswith(extension) and not f.startswith('.')]
            return sorted(files)
        except Exception as e:
            st.error(f"Error listing files in {dir_path}: {e}")
            logger_ui.error(f"Error listing files in {dir_path}: {e}", exc_info=True)
            return []
    else:
        logger_ui.warning(f"Directory not found: {dir_path}")
        return []

def get_captured_logs():
     log_stream.seek(0); logs = log_stream.read(); log_stream.truncate(0); log_stream.seek(0)
     new_log_entries = [entry for entry in logs.strip().split('\n') if entry]
     if new_log_entries:
         st.session_state.log_messages = new_log_entries + st.session_state.log_messages
         max_log_entries = 250
         if len(st.session_state.log_messages) > max_log_entries: st.session_state.log_messages = st.session_state.log_messages[:max_log_entries]

def update_dashboard_data():
    # This function now primarily populates the main incident store data
    # Filtering happens later before rendering tabs
    total_reports_checked_geo = 0; reports_with_coords = 0
    try: incidents = incident_store.get_all_incidents()
    except Exception as e: st.error(f"Failed to retrieve incidents: {e}"); return
    data = []; map_points = []; active_count = 0
    active_statuses = ["active", "updated", "received", "rcvd", "dispatched", "dsp", "acknowledged", "ack", "enroute", "enr", "onscene", "onscn", "monitoring"]
    for inc in incidents:
        try:
            is_active = inc.status and isinstance(inc.status, str) and inc.status.lower() in active_statuses
            if is_active: active_count += 1
            last_update_dt = pd.to_datetime(inc.last_updated_at, errors='coerce', utc=True)
            # Include ZIPs in the main dataframe now
            data.append({"ID": inc.incident_id[:8], "Type": inc.incident_type or "Unknown", "Status": inc.status or "Unknown", "Reports": inc.trend_data.get('report_count', 0), "Last Update": last_update_dt, "Summary": inc.summary or "N/A", "Actions": ", ".join(inc.recommended_actions) if inc.recommended_actions else "-", "Locations": len(inc.locations) if inc.locations else 0, "ZIPs": ", ".join(inc.zip_codes) or "N/A"})
            if inc.reports_core_data:
                 for report_core in inc.reports_core_data:
                     total_reports_checked_geo += 1
                     if report_core.coordinates and isinstance(report_core.coordinates, tuple) and len(report_core.coordinates) == 2:
                          try:
                              lat = float(report_core.coordinates[0]); lon = float(report_core.coordinates[1])
                              if -90 <= lat <= 90 and -180 <= lon <= 180: reports_with_coords += 1
                          except (ValueError, TypeError): pass
            if inc.locations:
                for lat, lon in inc.locations:
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and -90 <= lat <= 90 and -180 <= lon <= 180: map_points.append({'lat': lat, 'lon': lon, 'incident_id': inc.incident_id[:8], 'type': inc.incident_type or "Unknown", 'status': inc.status or "Unknown"}) # Add status for map tooltip
        except Exception as e: inc_id_log = getattr(inc, 'incident_id', 'UNKNOWN')[:8]; st.error(f"Error processing incident {inc_id_log}: {e}")
    st.session_state.incidents_df = pd.DataFrame(data) # This is the unfiltered dataframe
    st.session_state.total_incidents = len(incidents); st.session_state.active_incidents = active_count
    st.session_state.total_reports_geo_checked = total_reports_checked_geo; st.session_state.reports_with_geo = reports_with_coords
    # Create the base map data (unfiltered)
    if map_points:
         map_df = pd.DataFrame(map_points)
         if not map_df.empty:
            if 'lat' in map_df.columns and 'lon' in map_df.columns:
                map_df['lat'] = pd.to_numeric(map_df['lat'], errors='coerce'); map_df['lon'] = pd.to_numeric(map_df['lon'], errors='coerce')
                map_df.dropna(subset=['lat', 'lon'], inplace=True)
                if 'incident_id' in map_df.columns: map_df.drop_duplicates(subset=['lat', 'lon', 'incident_id'], inplace=True)
                else: map_df.drop_duplicates(subset=['lat', 'lon'], inplace=True)
            else: map_df = pd.DataFrame(columns=['lat', 'lon', 'incident_id', 'type', 'status'])
         st.session_state.map_data = map_df
    else: st.session_state.map_data = pd.DataFrame(columns=['lat', 'lon', 'incident_id', 'type', 'status'])
    get_captured_logs()


# --- UI Rendering ---

# Main Header
col_title, col_logo = st.columns([0.85, 0.15])
with col_title: st.title("🚨 EIDO Sentinel"); st.caption("AI-Powered Emergency Incident Processor")
# with col_logo: st.image("path/to/your/logo.png", width=80) # Optional logo
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/LXString/eido-sentinel/main/ui/logo.png", width=100)
    st.header("Agent Controls")
    st.divider()

    # --- Settings Expander ---
    with st.expander("⚙️ Configure Agent", expanded=False):
        # (Keep settings widgets as they were)
        st.subheader("LLM Configuration"); st.caption("Changes apply to this session only.")
        llm_provider_options = ['google', 'openrouter', 'local', 'none']
        llm_provider_index = llm_provider_options.index(st.session_state.llm_provider) if st.session_state.llm_provider in llm_provider_options else 0
        st.selectbox("☁️ LLM Provider:", options=llm_provider_options, index=llm_provider_index, key='llm_provider')
        if st.session_state.llm_provider == 'google':
            st.text_input("🔑 Google API Key:", key='google_api_key', type="password")
            google_model_options = st.session_state.get('google_model_options', [settings.google_model_name])
            current_google_model = st.session_state.get('google_model_name', settings.google_model_name)
            google_model_index = google_model_options.index(current_google_model) if current_google_model in google_model_options else 0
            st.selectbox("🧠 Google Model:", options=google_model_options, index=google_model_index, key='google_model_name')
        elif st.session_state.llm_provider == 'openrouter':
            st.text_input("🔑 OpenRouter API Key:", key='openrouter_api_key', type="password")
            st.text_input("🧠 OpenRouter Model:", key='openrouter_model_name')
            st.text_input("🔗 OpenRouter Base URL:", key='openrouter_api_base_url')
        elif st.session_state.llm_provider == 'local':
            st.text_input("🔗 Local LLM Base URL:", key='local_llm_api_base_url')
            st.text_input("🧠 Local LLM Model:", key='local_llm_model_name')
            st.text_input("🔑 Local LLM API Key:", key='local_llm_api_key', type="password")
        elif st.session_state.llm_provider == 'none': st.info("LLM features disabled.", icon="🚫")

    st.divider()

    # --- Ingestion Section ---
    st.header("📥 Data Ingestion")
    json_default = "" if st.session_state.clear_inputs_on_rerun else st.session_state.get('json_input_area', "")
    alert_default = "" if st.session_state.clear_inputs_on_rerun else st.session_state.get('alert_text_input_area', "")
    if st.session_state.clear_inputs_on_rerun: st.session_state.clear_inputs_on_rerun = False

    ingest_tab1, ingest_tab2 = st.tabs(["📄 EIDO JSON", "✍️ Raw Text"])
    with ingest_tab1:
        st.markdown("Upload EIDO Message files or paste JSON.")
        uploaded_files = st.file_uploader("📁 Upload File(s)", type="json", accept_multiple_files=True, key="file_uploader")
        json_input_area = st.text_area("📋 Paste JSON", value=json_default, height=150, key="json_input_area", placeholder='Paste EIDO Message JSON here...')
        st.markdown("---")
        st.markdown("**Load Sample:**")
        available_samples = list_files_in_dir(SAMPLE_DIR) # Use helper
        sample_options = ["-- Select Sample --"] + available_samples
        selected_sample = st.selectbox("📜 Select Sample EIDO:", options=sample_options, key="sample_select", index=0, label_visibility="collapsed")

    with ingest_tab2:
        st.markdown("Paste raw alert text for AI parsing.")
        alert_text_input_area = st.text_area("💬 Paste Alert Text", value=alert_default, height=200, key="alert_text_input_area", placeholder='ALERT: Vehicle collision at Main/Elm...')

    st.divider()

    # --- Process Button ---
    if st.button("🚀 Process Inputs", type="primary", use_container_width=True, help="Ingest and process the provided data."):
        # (Keep LLM config check)
        llm_needed = bool(alert_text_input_area)
        provider = st.session_state.get('llm_provider', 'none')
        key_missing = False
        if provider == 'google' and not st.session_state.get('google_api_key'): key_missing = True
        if provider == 'openrouter' and not st.session_state.get('openrouter_api_key'): key_missing = True
        if provider == 'local' and (not st.session_state.get('local_llm_api_base_url') or not st.session_state.get('local_llm_model_name')): key_missing = True

        if llm_needed and (provider == 'none' or key_missing):
            st.error(f"⚠️ LLM processing required for raw text is not configured.", icon="⚙️")
        else:
            with st.spinner('Processing inputs...'):
                # (Keep input gathering logic)
                reports_to_process_from_sources = []
                input_sources_names = []
                alert_text_to_process = None
                # ... (gather inputs) ...
                current_json_input = json_input_area
                if current_json_input:
                    try: reports_to_process_from_sources.append(json.loads(current_json_input)); input_sources_names.append("Pasted JSON")
                    except Exception as e: st.error(f"Pasted JSON Error: {e}")
                elif selected_sample and selected_sample != "-- Select Sample --":
                    try:
                        with open(os.path.join(SAMPLE_DIR, selected_sample), 'r', encoding='utf-8') as f: reports_to_process_from_sources.append(json.load(f)); input_sources_names.append(f"Sample: {selected_sample}")
                        logger_ui.info(f"Processing sample file: {selected_sample}") # Log sample processing attempt
                    except Exception as e: st.error(f"Sample Error loading {selected_sample}: {e}"); logger_ui.error(f"Sample loading error: {e}", exc_info=True)
                if uploaded_files:
                    for uf in uploaded_files:
                        try: reports_to_process_from_sources.append(json.loads(uf.getvalue().decode("utf-8"))); input_sources_names.append(f"File: {uf.name}")
                        except Exception as e: st.error(f"File Error ({uf.name}): {e}")
                current_alert_text = alert_text_input_area
                if current_alert_text: alert_text_to_process = current_alert_text.strip()

                if not reports_to_process_from_sources and not alert_text_to_process:
                    st.warning("No data provided for processing.")
                else:
                    # (Keep the rest of the processing loop: JSON processing, Text processing, Summary display)
                    total_processed_count = 0; total_error_count = 0
                    total_start_time = time.time(); status_messages = []
                    # --- Process EIDO JSON ---
                    if reports_to_process_from_sources:
                        final_reports_list = []; error_count_loading = 0
                        for idx, loaded_data in enumerate(reports_to_process_from_sources): # Expand lists
                            source_name = input_sources_names[idx] if idx < len(input_sources_names) else "Unknown"
                            if isinstance(loaded_data, list):
                                for item_index, item in enumerate(loaded_data):
                                    if isinstance(item, dict): final_reports_list.append(item)
                                    else: logger_ui.warning(f"Skipping non-dict item #{item_index+1} in list from '{source_name}'."); error_count_loading += 1
                            elif isinstance(loaded_data, dict): final_reports_list.append(loaded_data)
                            else: logger_ui.error(f"Skipping invalid data type from '{source_name}': {type(loaded_data)}."); error_count_loading += 1
                        num_reports_to_process = len(final_reports_list)
                        if num_reports_to_process > 0: # Process individual reports
                            processed_count_json = 0; error_count_json = 0
                            for i, report_dict in enumerate(final_reports_list):
                                msg_id_for_log = report_dict.get('eidoMessageIdentifier', report_dict.get('$id', f"json_{i+1}"))[:20]
                                try:
                                    result_dict = eido_agent_instance.process_report_json(report_dict)
                                    if result_dict.get('status', '').lower() == 'success': processed_count_json += 1
                                    else: error_count_json += 1; status_messages.append(f"EIDO Err ({msg_id_for_log}...): {result_dict.get('status')}")
                                except Exception as e: error_count_json += 1; status_messages.append(f"EIDO Crit Err ({msg_id_for_log}...): {e}"); logger_ui.critical(f"EIDO processing error: {e}", exc_info=True)
                            total_processed_count += processed_count_json
                            total_error_count += error_count_json + error_count_loading
                            status_messages.append(f"EIDO JSON: {processed_count_json}/{num_reports_to_process} processed, {error_count_json + error_count_loading} errors/skipped.")

                    # --- Process Raw Alert Text ---
                    if alert_text_to_process:
                         # (Keep text processing loop as is - it handles list output)
                         processed_count_text = 0; error_count_text = 0
                         try:
                             list_of_results = eido_agent_instance.process_alert_text(alert_text_to_process)
                             if not list_of_results: error_count_text += 1; status_messages.append("Alert Text Err: Agent returned no results.")
                             else:
                                 num_events = len(list_of_results)
                                 for idx, result_dict in enumerate(list_of_results):
                                     event_num = idx + 1; status_msg = result_dict.get('status', 'Unknown'); msg_id = result_dict.get('message_id', f'event_{event_num}')[:20]
                                     if status_msg.lower() == 'success': processed_count_text += 1
                                     else: error_count_text += 1; status_messages.append(f"Alert Event {event_num}/{num_events} Err ({msg_id}...): {status_msg}")
                                 status_messages.append(f"Alert Text Block: {processed_count_text}/{num_events} events processed, {error_count_text} errors.")
                         except Exception as e: error_count_text += 1; status_messages.append(f"Alert Text Crit Err: {e}"); logger_ui.critical("Alert text processing block error", exc_info=True)
                         total_processed_count += processed_count_text
                         total_error_count += error_count_text

                    # --- Display Summary ---
                    total_end_time = time.time(); total_duration = total_end_time - total_start_time
                    if total_error_count > 0: st.warning(f"Processing finished: {total_processed_count} succeeded, {total_error_count} failed ({total_duration:.2f}s).", icon="⚠️")
                    else: st.success(f"Processing finished: {total_processed_count} succeeded ({total_duration:.2f}s).", icon="✅")
                    if status_messages:
                        with st.expander("Show Processing Status Details"):
                            for msg in status_messages:
                                if "Err" in msg or "skipped" in msg: st.warning(msg)
                                else: st.info(msg)

                    update_dashboard_data()
                    get_captured_logs()
                    st.session_state.clear_inputs_on_rerun = True
                    time.sleep(0.1); st.rerun()

    st.divider()
    # --- Admin Actions Expander ---
    with st.expander("⚠️ Admin Actions", expanded=False):
        st.caption("Actions here affect the current session's data.")
        if st.button("🗑️ Clear All Incidents", key="clear_button", use_container_width=True):
             try:
                 count = len(incident_store.get_all_incidents())
                 if count > 0:
                      incident_store.clear_store()
                      keys_to_reset = ['incidents_df', 'map_data', 'total_incidents', 'active_incidents', 'total_reports_geo_checked', 'reports_with_geo', 'filtered_incidents']
                      for key in keys_to_reset: init_session_state() # Re-init to defaults
                      update_dashboard_data() # Update base data
                      st.success(f"Cleared {count} incidents.")
                      time.sleep(0.5); st.rerun()
                 else: st.info("Incident store is already empty.")
             except Exception as e: st.error(f"Failed to clear store: {e}")
             get_captured_logs()

    st.divider()
    # --- Log Area Expander ---
    with st.expander("📄 Processing Log", expanded=False):
        get_captured_logs()
        log_container = st.container(height=300)
        with log_container:
            if st.session_state.log_messages: st.code("\n".join(st.session_state.log_messages), language='log')
            else: st.caption("Log is empty.")


# --- Main Dashboard Area ---
dashboard = st.container()
with dashboard:
    st.header("📊 Incident Dashboard")
    # --- Key Metrics Row ---
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("📈 Total Incidents", st.session_state.total_incidents)
    metric_col2.metric("🔥 Active Incidents", st.session_state.active_incidents)
    try: avg_reports = st.session_state.incidents_df['Reports'].mean() if not st.session_state.incidents_df.empty and 'Reports' in st.session_state.incidents_df.columns and st.session_state.incidents_df['Reports'].notna().any() else 0
    except Exception: avg_reports = 0
    metric_col3.metric("📄 Avg Reports/Incident", f"{avg_reports:.1f}")
    total_geo = st.session_state.get('total_reports_geo_checked', 0); found_geo = st.session_state.get('reports_with_geo', 0)
    geo_perc = (found_geo / total_geo * 100) if total_geo > 0 else 0
    metric_col4.metric("📍 Reports w/ Coords", f"{found_geo}/{total_geo}", f"{geo_perc:.1f}%")
    st.divider()

    # --- Dashboard Controls / Filters ---
    st.subheader("🔎 Filter & Analyze Incidents")
    filter_col1, filter_col2, filter_col3 = st.columns([0.4, 0.3, 0.3])

    # Prepare filter options from the complete (unfiltered) data
    all_incidents_for_filter = incident_store.get_all_incidents() # Get fresh list
    available_types = sorted(list(set(inc.incident_type for inc in all_incidents_for_filter if inc.incident_type)))
    available_statuses = sorted(list(set(inc.status for inc in all_incidents_for_filter if inc.status)))
    available_zips = sorted(list(set(zip_code for inc in all_incidents_for_filter for zip_code in inc.zip_codes if zip_code))) # Filter out None/empty zips

    with filter_col1:
        selected_types = st.multiselect("Filter by Type:", options=available_types, default=[], key="filter_type")
    with filter_col2:
        selected_statuses = st.multiselect("Filter by Status:", options=available_statuses, default=[], key="filter_status")
    with filter_col3:
        selected_zips = st.multiselect("Filter by ZIP Code:", options=available_zips, default=[], key="filter_zip")

    # --- Apply Filters ---
    # Start with all incidents and filter down
    filtered_incidents = all_incidents_for_filter
    if selected_types:
        filtered_incidents = [inc for inc in filtered_incidents if inc.incident_type in selected_types]
    if selected_statuses:
        filtered_incidents = [inc for inc in filtered_incidents if inc.status in selected_statuses]
    if selected_zips:
        filtered_incidents = [inc for inc in filtered_incidents if any(zip_code in selected_zips for zip_code in inc.zip_codes)]

    # Store filtered list in session state for reuse in other tabs
    st.session_state.filtered_incidents = filtered_incidents

    # Create DataFrames from filtered data for display in tabs
    filtered_data = []
    filtered_map_points = []
    for inc in st.session_state.filtered_incidents: # Use the filtered list from state
        last_update_dt = pd.to_datetime(inc.last_updated_at, errors='coerce', utc=True)
        filtered_data.append({"ID": inc.incident_id[:8], "Type": inc.incident_type or "Unknown", "Status": inc.status or "Unknown", "Reports": inc.trend_data.get('report_count', 0), "Last Update": last_update_dt, "Summary": inc.summary or "N/A", "Locations": len(inc.locations), "ZIPs": ", ".join(inc.zip_codes or [])})
        if inc.locations:
            for lat, lon in inc.locations:
                 if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    filtered_map_points.append({'lat': lat, 'lon': lon, 'incident_id': inc.incident_id[:8], 'type': inc.incident_type or "Unknown", 'status': inc.status or "Unknown"})

    filtered_df = pd.DataFrame(filtered_data)
    filtered_map_df = pd.DataFrame(filtered_map_points)
    if not filtered_map_df.empty:
        filtered_map_df['lat'] = pd.to_numeric(filtered_map_df['lat'], errors='coerce')
        filtered_map_df['lon'] = pd.to_numeric(filtered_map_df['lon'], errors='coerce')
        filtered_map_df.dropna(subset=['lat', 'lon'], inplace=True)

    st.divider()

    # --- Main Content Tabs (Add EIDO Generator) ---
    st.subheader("📊 Filtered Views")
    tab_list, tab_map, tab_charts, tab_details, tab_warning, tab_eido_explorer, tab_eido_generator = st.tabs([
        "🗓️ **List**", "🗺️ **Map**", "📈 **Charts**", "🔍 **Details**", "📢 **Warnings**", "📄 **EIDO Explorer**", "📝 **EIDO Generator**"
    ])

    # --- Incident List Tab (Filtered) ---
    with tab_list:
        st.caption(f"Displaying {len(st.session_state.filtered_incidents)} incidents based on filters.") # Use count from state
        if not filtered_df.empty:
            # (Keep dataframe display logic using filtered_df)
            df_display = filtered_df.copy()
            df_display.sort_values(by="Last Update", ascending=False, inplace=True, na_position='last')
            st.dataframe(df_display, use_container_width=True, hide_index=True, column_order=("ID", "Type", "Status", "Last Update", "Reports", "Locations", "ZIPs", "Summary"), column_config={ #... keep config ...
                 "ID": st.column_config.TextColumn("ID", width="small", disabled=True), "Type": st.column_config.TextColumn("Type", width="medium"), "Status": st.column_config.TextColumn("Status", width="small"), "Last Update": st.column_config.DatetimeColumn("Last Update", format="YYYY-MM-DD HH:mm", width="small"), "Reports": st.column_config.NumberColumn("Reports", format="%d", width="small"), "Locations": st.column_config.NumberColumn("Locs", format="%d", width="small"), "ZIPs": st.column_config.TextColumn("ZIPs", width="small"), "Summary": st.column_config.TextColumn("Summary", width="large")
            })
        else: st.info("No incidents match the current filter criteria.", icon="🚫")

    # --- Geographic Map Tab (PyDeck) ---
    with tab_map:
        st.caption(f"Displaying locations for {len(st.session_state.filtered_incidents)} filtered incidents.")
        if not filtered_map_df.empty:
            # (Keep PyDeck map logic using filtered_map_df)
            try:
                mid_lat = filtered_map_df['lat'].median(); mid_lon = filtered_map_df['lon'].median()
                view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=10, pitch=45)
                scatter_layer = pdk.Layer('ScatterplotLayer', data=filtered_map_df, get_position='[lon, lat]', get_color='[200, 30, 0, 160]', get_radius=50, pickable=True, auto_highlight=True)
                heatmap_layer = pdk.Layer("HeatmapLayer", data=filtered_map_df, opacity=0.7, get_position=["lon", "lat"], aggregation=pdk.types.String("MEAN"), threshold=0.1, get_weight=1, pickable=True)
                tooltip = {"html": "<b>Incident:</b> {incident_id}<br/><b>Type:</b> {type}<br/><b>Status:</b> {status}", "style": {"backgroundColor": "steelblue", "color": "white"}}
                st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v10', initial_view_state=view_state, layers=[heatmap_layer, scatter_layer], tooltip=tooltip))
                with st.expander("Show Raw Map Data"): st.dataframe(filtered_map_df[['incident_id', 'type', 'lat', 'lon']].round(6), use_container_width=True, hide_index=True)
            except Exception as map_error:
                 st.error(f"Error displaying PyDeck map: {map_error}")
                 # (Keep fallback to st.map)
        else: st.info("No geocoded locations match the current filter criteria.", icon="🗺️")

    # --- Charts Tab ---
    with tab_charts:
        st.caption(f"Displaying charts for {len(st.session_state.filtered_incidents)} filtered incidents.")
        if not filtered_df.empty:
            # (Keep charts logic using filtered_df)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("##### Status Distribution")
                status_counts = filtered_df['Status'].value_counts()
                if not status_counts.empty: st.bar_chart(status_counts)
                else: st.caption("No status data.")
            with chart_col2:
                st.markdown("##### Top Incident Types")
                type_counts = filtered_df['Type'].value_counts().head(10)
                if not type_counts.empty: st.bar_chart(type_counts)
                else: st.caption("No type data.")
        else: st.info("No incidents match filters to display charts.", icon="📊")

    # --- Details View Tab ---
    with tab_details:
        st.caption(f"Select one of the {len(st.session_state.filtered_incidents)} filtered incidents for details.")
        # Use filtered incidents from state
        filtered_incidents_map = {inc.incident_id[:8]: inc.incident_id for inc in st.session_state.filtered_incidents}
        available_short_ids_filtered = sorted(list(filtered_incidents_map.keys()), reverse=True)
        if available_short_ids_filtered:
            selected_id_short_detail = st.selectbox("Select Filtered Incident ID:", options=["-- Select --"] + available_short_ids_filtered, index=0, key="detail_incident_selector_filtered")
            if selected_id_short_detail and selected_id_short_detail != "-- Select --":
                full_incident_id_detail = filtered_incidents_map.get(selected_id_short_detail)
                selected_incident_obj_detail = incident_store.get_incident(full_incident_id_detail) # Get full object
                if selected_incident_obj_detail:
                    # (Keep the detailed display logic for summary, actions, reports, original JSON)
                    st.markdown(f"#### Incident `{selected_id_short_detail}` Details"); st.divider()
                    info_col1, info_col2 = st.columns(2) # ... display info ...
                    with info_col1: st.markdown(f"**Type:** {selected_incident_obj_detail.incident_type or 'N/A'}"); st.markdown(f"**Status:** `{selected_incident_obj_detail.status or 'Unknown'}`"); st.markdown(f"**Reports:** {selected_incident_obj_detail.trend_data.get('report_count', 0)}")
                    with info_col2: created_dt = pd.to_datetime(selected_incident_obj_detail.created_at, utc=True, errors='coerce'); updated_dt = pd.to_datetime(selected_incident_obj_detail.last_updated_at, utc=True, errors='coerce'); st.markdown(f"**Created:** {created_dt.strftime('%Y-%m-%d %H:%M') if pd.notna(created_dt) else 'N/A'} UTC"); st.markdown(f"**Updated:** {updated_dt.strftime('%Y-%m-%d %H:%M') if pd.notna(updated_dt) else 'N/A'} UTC"); st.markdown(f"**Last Match:** `{selected_incident_obj_detail.trend_data.get('match_info', 'N/A')}`")
                    st.divider(); st.markdown("##### 📄 AI Summary"); st.info(selected_incident_obj_detail.summary or "_No summary generated._", icon="🤖")
                    st.markdown("##### ✅ Recommended Actions") # ... display actions ...
                    if selected_incident_obj_detail.recommended_actions: st.markdown("\n".join(f"- {action}" for action in selected_incident_obj_detail.recommended_actions))
                    else: st.caption("_No actions recommended._")
                    st.divider()
                    with st.expander(f"📜 Associated Reports ({len(selected_incident_obj_detail.reports_core_data or [])})", expanded=False): # ... display reports ...
                         if selected_incident_obj_detail.reports_core_data: # ... display dataframe ...
                            report_data_display = [] # ... build dataframe ...
                            st.dataframe(pd.DataFrame(report_data_display), use_container_width=True, hide_index=True, column_config={ #... keep config ...
                            })
                         else: st.info("No report data associated.")
                    with st.expander("📄 Original EIDO JSON Data", expanded=False): # ... display JSON ...
                         if selected_incident_obj_detail.reports_core_data: # ... display JSON text area + download ...
                            original_eido_dicts = [] # ... build list ...
                            if original_eido_dicts: # ... display ace editor + download ...
                               pass
                            else: st.info("No original EIDO JSON found.")
                         else: st.info("No report data associated.")
                else: st.warning(f"Could not retrieve details for incident ID {selected_id_short_detail}.")
            elif available_short_ids_filtered: st.info("Select a filtered Incident ID to view details.", icon="👆")
        else: st.info("No incidents match filters to show details.", icon="🚫")

    # --- Warnings Tab ---
    with tab_warning:
        st.subheader("📢 Generate Incident Warning Text")
        st.caption(f"Based on the {len(st.session_state.filtered_incidents)} currently filtered incidents.")
        if st.session_state.filtered_incidents:
            # (Keep warning generation logic using st.session_state.filtered_incidents)
            warning_level = st.select_slider("Select Warning Severity:", options=["Informational", "Advisory", "Watch", "Warning"], value="Advisory")
            custom_message = st.text_area("Add Custom Message (Optional):", height=100)
            if st.button("📝 Generate Warning", use_container_width=True):
                warning_text = f"**--- {warning_level.upper()} ---**\n\n" # ... build warning text ...
                st.text_area("Generated Warning Text (Copy below):", value=warning_text, height=300, key="warning_output")
        else: st.info("Apply filters to select incidents for warning generation.", icon="⚠️")

    # --- EIDO Explorer Tab ---
    with tab_eido_explorer:
        st.subheader("📄 Explore Original EIDO Data")
        st.caption("View the raw EIDO JSON associated with processed reports.")
        # Use filtered incidents from state
        incidents_map_explorer = {inc.incident_id[:8]: inc.incident_id for inc in st.session_state.filtered_incidents}
        available_ids_explorer = sorted(list(incidents_map_explorer.keys()), reverse=True)
        if not available_ids_explorer:
            st.info("No incidents match filters to explore EIDO data.", icon="🚫")
        else:
            selected_inc_id_short_exp = st.selectbox("Select Incident to Explore:", options=["-- Select Incident --"] + available_ids_explorer, index=0, key="eido_explorer_inc_select")
            if selected_inc_id_short_exp != "-- Select Incident --":
                selected_inc_full_id_exp = incidents_map_explorer.get(selected_inc_id_short_exp)
                selected_inc_obj_exp = incident_store.get_incident(selected_inc_full_id_exp)
                if selected_inc_obj_exp and selected_inc_obj_exp.reports_core_data:
                    # (Keep report selector and Ace editor display logic)
                    report_options = {} # ... build options ...
                    if not report_options: st.warning("Selected incident has no associated reports.")
                    else:
                        selected_report_label = st.selectbox(f"Select Report for Incident {selected_inc_id_short_exp}:", options=["-- Select Report --"] + list(report_options.keys()), index=0, key=f"eido_explorer_report_select_{selected_inc_id_short_exp}")
                        if selected_report_label != "-- Select Report --":
                            selected_report_id = report_options.get(selected_report_label)
                            selected_report_core = next((rc for rc in selected_inc_obj_exp.reports_core_data if rc.report_id == selected_report_id), None)
                            if selected_report_core and selected_report_core.original_eido_dict:
                                # ... display ace editor + download ...
                                eido_str_display = json.dumps(selected_report_core.original_eido_dict, indent=2)
                                st_ace(value=eido_str_display, language="json", theme="github", readonly=True, key=f"ace_editor_{selected_report_id}", height=400, wrap=True)
                                st.download_button(label="📥 Download this EIDO JSON", data=eido_str_display.encode('utf-8'), file_name=f"report_{selected_report_core.report_id[:8]}_eido.json", mime="application/json", key=f"dl_eido_report_{selected_report_id}")
                            elif selected_report_core: st.warning("Selected report does not have original EIDO data stored.")
                elif selected_inc_obj_exp: st.info("Selected incident has no associated report data.")

    # --- NEW: EIDO Generator Tab ---
    with tab_eido_generator:
        st.subheader("📝 Generate Compliant EIDO JSON")
        st.info("Use LLM assistance to fill a standard EIDO template based on a scenario description.", icon="🤖")

        # Select Template
        available_templates = list_files_in_dir(TEMPLATE_DIR)
        if not available_templates:
            st.warning(f"No EIDO templates found in `{TEMPLATE_DIR}`. Please create template files (e.g., `traffic_collision.json`) with placeholders like `[PLACEHOLDER]`.", icon="⚠️")
        else:
            template_options = ["-- Select Template --"] + available_templates
            selected_template_file = st.selectbox(
                "Select EIDO Template:",
                options=template_options, index=0, key="generator_template_select"
            )

            # Scenario Input
            scenario_description = st.text_area(
                "Enter Scenario Description:", height=150, key="generator_scenario_input",
                placeholder="Describe the incident (e.g., 'Structure fire at 100 Main St, Apt 5, reported by Engine 3 at 08:15 UTC...')"
            )

            # Generate Button
            if st.button("✨ Generate EIDO from Template", key="generator_button", disabled=(selected_template_file == "-- Select Template --" or not scenario_description)):
                if selected_template_file != "-- Select Template --" and scenario_description:
                    template_path = os.path.join(TEMPLATE_DIR, selected_template_file)
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            template_content = f.read()

                        # Check LLM config (needed for generation)
                        provider = st.session_state.get('llm_provider', 'none')
                        key_missing = False
                        if provider == 'google' and not st.session_state.get('google_api_key'): key_missing = True
                        if provider == 'openrouter' and not st.session_state.get('openrouter_api_key'): key_missing = True
                        if provider == 'local' and (not st.session_state.get('local_llm_api_base_url') or not st.session_state.get('local_llm_model_name')): key_missing = True

                        if provider == 'none' or key_missing:
                             st.error("LLM is not configured. Please check API keys/models in the Agent Settings.", icon="⚙️")
                        else:
                            with st.spinner("Generating EIDO JSON..."):
                                generated_json_str = fill_eido_template(template_content, scenario_description)
                                if generated_json_str:
                                    st.session_state.generated_eido_json = generated_json_str # Store result in session state
                                    st.success("EIDO JSON generated successfully!", icon="✅")
                                else:
                                    st.session_state.generated_eido_json = None
                                    st.error("Failed to generate EIDO JSON from template using LLM.", icon="❌")
                                get_captured_logs() # Capture logs from generation attempt

                    except FileNotFoundError:
                        st.error(f"Template file not found: {selected_template_file}")
                    except Exception as e:
                        st.error(f"Error loading template or generating EIDO: {e}")
                        logger_ui.error(f"EIDO Generator error: {e}", exc_info=True)
                        st.session_state.generated_eido_json = None

            # Display Generated EIDO
            if st.session_state.generated_eido_json:
                st.markdown("---")
                st.markdown("**Generated EIDO JSON:**")
                try:
                    # Display using Ace editor
                    st_ace(
                        value=st.session_state.generated_eido_json,
                        language="json", theme="github", keybinding="vscode",
                        font_size=12, height=400, show_gutter=True, show_print_margin=False,
                        wrap=True, auto_update=False, readonly=True, key="ace_editor_generator_output"
                    )
                    # Download button
                    st.download_button(
                        label="📥 Download Generated EIDO",
                        data=st.session_state.generated_eido_json.encode('utf-8'),
                        file_name=f"generated_{selected_template_file.replace('.json','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                        mime="application/json",
                        key="dl_generated_eido"
                    )
                except Exception as display_err:
                    st.error(f"Error displaying generated EIDO: {display_err}")
                    st.json(st.session_state.generated_eido_json) # Fallback


# --- Footer ---
st.divider()
st.caption(f"EIDO Sentinel v0.8.0 | AI Incident Processor POC") # Version bump

# Final log capture
get_captured_logs()