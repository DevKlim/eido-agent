# ui/app.py
import streamlit as st
import json
import os
import pandas as pd
import time
from datetime import datetime, timezone # Import timezone explicitly
import sys
import uuid
import logging # Import logging

st.set_page_config(
    layout="wide",
    page_title="EIDO Sentinel | AI Incident Processor",
    page_icon="🚨"
)

# --- Setup Python Path ---
# Ensure this path is correct relative to where you run streamlit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Attempt Custom Module Imports ---
modules_imported_successfully = True
import_error_message = ""
original_error = None # Store the original error

try:
    # Import settings FIRST
    from config.settings import settings
    # Configure logging AFTER settings are loaded
    # Use force=True to override any Streamlit default handlers if necessary
    logging.basicConfig(
        level=settings.log_level.upper(), # Use validated log level
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logger_ui = logging.getLogger(__name__) # Get logger for this module
    logger_ui.info("Logging configured successfully from ui/app.py.")
    logger_ui.info(f"Log level set to: {settings.log_level.upper()}")

    # Now import other modules
    from agent.agent_core import eido_agent_instance
    from services.storage import incident_store
    from data_models.schemas import Incident # Keep Incident import

except ImportError as e:
    modules_imported_successfully = False
    import_error_message = f"Import Error: {e}"
    original_error = e
except AttributeError as e:
     modules_imported_successfully = False
     import_error_message = f"Attribute Error during setup (check settings/imports): {e}"
     original_error = e
except Exception as e:
     modules_imported_successfully = False
     import_error_message = f"Unexpected error during setup: {e}"
     original_error = e
     # Log critical error if logger is available
     if 'logger_ui' in locals():
         logger_ui.critical(f"CRITICAL SETUP ERROR: {e}", exc_info=True)
     else:
         print(f"CRITICAL SETUP ERROR (Logger not available): {e}")


# --- Check Imports and Halt if Necessary ---
if not modules_imported_successfully:
    st.error(f"🚨 CRITICAL ERROR: Failed during application setup.")
    st.error(f"Details: {import_error_message}")
    if original_error:
        st.exception(original_error)
    st.warning("Please ensure:")
    st.markdown("- You are running the app from the project's root directory (`eido-sentinel/`).")
    st.markdown("- You have installed all dependencies (`./install_dependencies.sh` or `pip install -r requirements.txt`).")
    st.markdown("- The project structure matches the `README.md`.")
    st.markdown("- Python files (`agent/agent_core.py`, `services/storage.py`, `data_models/schemas.py`, `agent/llm_interface.py`, `config/settings.py`, `agent/alert_parser.py` etc.) are present and syntactically correct.")
    st.info("Expected command: `streamlit run ui/app.py`")
    st.stop() # Stop execution if core components can't be loaded


# --- In-memory handler to capture logs for UI display ---
from io import StringIO
import logging.handlers

log_stream = StringIO()
# Use a specific formatter for the UI log display
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setFormatter(log_formatter)

# Add handler to root logger to capture everything
# Check if handler already added to prevent duplicates on rerun
root_logger = logging.getLogger()
# Set root logger level based on settings to ensure messages are passed to handlers
root_logger.setLevel(settings.log_level.upper())
if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == log_stream for h in root_logger.handlers):
    root_logger.addHandler(stream_handler)
    logger_ui.debug("Log capture stream handler added to root logger.")


# --- Global Variables & Constants ---
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sample_eido'))
if not os.path.exists(SAMPLE_DIR):
    try:
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        logger_ui.info(f"Created sample directory at {SAMPLE_DIR}")
    except Exception as e:
        st.error(f"Failed to create sample directory {SAMPLE_DIR}: {e}")
        logger_ui.error(f"Failed to create sample directory {SAMPLE_DIR}: {e}", exc_info=True)

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'incidents_df': pd.DataFrame(columns=["ID", "Type", "Status", "Reports", "Last Update", "Summary", "Actions", "Locations"]),
        'log_messages': [],
        'map_data': pd.DataFrame(columns=['lat', 'lon', 'incident_id', 'type']),
        'total_incidents': 0,
        'active_incidents': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Helper Functions ---
def list_sample_files():
    """Lists JSON files in the sample directory."""
    if os.path.exists(SAMPLE_DIR):
        try:
            files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith('.json') and not f.startswith('.')]
            return sorted(files)
        except Exception as e:
            st.error(f"Error listing sample files in {SAMPLE_DIR}: {e}")
            logger_ui.error(f"Error listing sample files in {SAMPLE_DIR}: {e}", exc_info=True)
            return []
    return []

def get_captured_logs():
     """Retrieves logs captured by the stream handler and updates session state."""
     log_stream.seek(0)
     logs = log_stream.read()
     # Reset stream buffer for next logs
     log_stream.truncate(0)
     log_stream.seek(0)
     # Prepend new logs to session state list
     new_log_entries = logs.strip().split('\n')
     new_log_entries = [entry for entry in new_log_entries if entry] # Remove empty strings
     # Update session state only if there are new logs
     if new_log_entries:
         st.session_state.log_messages = new_log_entries + st.session_state.log_messages
         # Trim log history
         max_log_entries = 200
         if len(st.session_state.log_messages) > max_log_entries:
            st.session_state.log_messages = st.session_state.log_messages[:max_log_entries]

def update_dashboard_data():
    """Updates the Streamlit dataframe and map data from the incident store."""
    try:
        incidents = incident_store.get_all_incidents()
        logger_ui.debug(f"Updating dashboard with {len(incidents)} incidents from store.")
    except Exception as e:
        st.error(f"Failed to retrieve incidents from store: {e}")
        logger_ui.error(f"Error accessing incident store: {e}", exc_info=True)
        return

    data = []
    map_points = []
    active_count = 0
    # Use the same definition as in matching.py or a robust status check
    active_statuses = [
        "active", "updated", "received", "rcvd",
        "dispatched", "dsp", "acknowledged", "ack",
        "enroute", "enr", "onscene", "onscn",
        "monitoring" # Added monitoring as active
    ]

    for inc in incidents:
        try:
            # Make status check robust and case-insensitive
            is_active = inc.status and isinstance(inc.status, str) and inc.status.lower() in active_statuses
            if is_active:
                active_count += 1

            # Ensure timezone aware for consistency
            last_update_dt = pd.to_datetime(inc.last_updated_at, errors='coerce', utc=True)

            data.append({
                "ID": inc.incident_id[:8],
                "Type": inc.incident_type or "Unknown",
                "Status": inc.status or "Unknown",
                "Reports": inc.trend_data.get('report_count', 0),
                "Last Update": last_update_dt, # Keep as datetime
                "Summary": inc.summary or "N/A",
                "Actions": ", ".join(inc.recommended_actions) if inc.recommended_actions else "-",
                "Locations": len(inc.locations) if inc.locations else 0
            })

            if inc.locations:
                for lat, lon in inc.locations:
                    # Validate coordinates before adding
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and -90 <= lat <= 90 and -180 <= lon <= 180:
                        map_points.append({
                            'lat': lat,
                            'lon': lon,
                            'incident_id': inc.incident_id[:8],
                            'type': inc.incident_type or "Unknown"
                        })
                    else:
                         logger_ui.warning(f"Invalid coordinates ({lat}, {lon}) found for incident {inc.incident_id[:8]} - skipping map point.")

        except Exception as e:
            inc_id_log = getattr(inc, 'incident_id', 'UNKNOWN')[:8]
            st.error(f"Error processing incident {inc_id_log} for dashboard: {e}")
            logger_ui.error(f"Dashboard update error for incident {inc_id_log}: {e}", exc_info=True)
            continue

    st.session_state.incidents_df = pd.DataFrame(data)
    st.session_state.total_incidents = len(incidents)
    st.session_state.active_incidents = active_count

    if map_points:
         map_df = pd.DataFrame(map_points)
         # Check for duplicates before dropping
         if not map_df.empty:
            map_df.drop_duplicates(subset=['lat', 'lon', 'incident_id'], inplace=True)
            map_df['lat'] = pd.to_numeric(map_df['lat'], errors='coerce')
            map_df['lon'] = pd.to_numeric(map_df['lon'], errors='coerce')
            map_df.dropna(subset=['lat', 'lon'], inplace=True)
         st.session_state.map_data = map_df
    else:
         st.session_state.map_data = pd.DataFrame(columns=['lat', 'lon', 'incident_id', 'type'])

    # Capture logs generated during the update
    get_captured_logs()


# --- UI Rendering ---

# Header Section
st.title("🚨 EIDO Sentinel")
st.caption("AI-Powered Emergency Incident Processor")

st.divider()

# Main Layout: Input Sidebar + Dashboard Area
sidebar = st.sidebar
dashboard = st.container()

# --- Sidebar for Inputs ---
with sidebar:
    st.header("📥 Report Ingestion")

    # --- Input Method Tabs ---
    input_tab1, input_tab2 = st.tabs(["EIDO JSON Input", "Raw Alert Text Input"])

    with input_tab1:
        st.markdown("Provide EIDO **Messages** (JSON format) via file upload, text input, or load samples.")
        uploaded_files = st.file_uploader(
            "Upload EIDO Message file(s)", type="json", accept_multiple_files=True, key="file_uploader",
            help="Upload files containing single EIDO Messages (JSON object) or lists of Messages (JSON array)."
        )
        json_input_area = st.text_area(
            "Paste EIDO Message JSON", height=150, key="json_input_area",
            placeholder='{\n  "eidoMessageIdentifier": "...",\n  "incidentComponent": [ ... ],\n  ...\n}',
            help="Paste a single EIDO Message object or a list of Message objects."
        )
        st.markdown("---")
        st.subheader("Sample Reports")
        available_samples = list_sample_files()
        if available_samples:
            sample_options = ["-- Select a Sample --"] + available_samples
            selected_sample = st.selectbox(
                "Load a sample EIDO Message:", options=sample_options, key="sample_select", index=0,
                help="Select a sample file conforming to the EidoMessage schema."
            )
        else:
            st.info("No sample files found in 'sample_eido/'. Ensure samples use the EidoMessage structure.")
            selected_sample = None

    with input_tab2:
        st.markdown("Paste raw alert text (e.g., from CAD, SMS, transcript snippet) below.")
        alert_text_input_area = st.text_area(
            "Paste Raw Alert Text", height=200, key="alert_text_input_area",
            placeholder='Example:\nALERT: Vehicle collision reported at Main St / Elm Ave around 3:30 PM...\nIncident # CAD-2024-98765...',
            help="Enter unstructured or semi-structured alert text for the AI to parse."
        )

    st.markdown("---") # Separator before the button

    if st.button("🚀 Process Inputs", type="primary", use_container_width=True):
        reports_to_process_from_sources = [] # Holds loaded EIDO JSON data (dict or list)
        input_sources_names = [] # Tracks EIDO JSON source names
        alert_text_to_process = None # Holds raw alert text

        # 1. Gather EIDO JSON data (from files, text area, samples)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    string_data = uploaded_file.getvalue().decode("utf-8")
                    loaded_data = json.loads(string_data)
                    reports_to_process_from_sources.append(loaded_data)
                    input_sources_names.append(f"File: {uploaded_file.name}")
                    logger_ui.debug(f"Loaded EIDO JSON data from file: {uploaded_file.name}")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON in {uploaded_file.name}: {e}")
                    logger_ui.error(f"Invalid JSON in {uploaded_file.name}: {e}")
                except Exception as e:
                    st.error(f"Error reading {uploaded_file.name}: {e}")
                    logger_ui.error(f"Failed reading file {uploaded_file.name}: {e}", exc_info=True)

        if json_input_area:
            try:
                loaded_data = json.loads(json_input_area)
                reports_to_process_from_sources.append(loaded_data)
                input_sources_names.append("Pasted EIDO JSON")
                logger_ui.debug("Loaded EIDO JSON data from pasted text.")
            except json.JSONDecodeError as e:
                 st.error(f"Invalid JSON in EIDO text area: {e}")
                 logger_ui.error(f"Processing pasted EIDO JSON failed: {e}")
            except Exception as e:
                 st.error(f"Error processing pasted EIDO JSON: {e}")
                 logger_ui.error(f"Error processing pasted EIDO JSON: {e}", exc_info=True)

        elif selected_sample and selected_sample != "-- Select a Sample --":
            try:
                sample_path = os.path.join(SAMPLE_DIR, selected_sample)
                with open(sample_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    reports_to_process_from_sources.append(loaded_data)
                    input_sources_names.append(f"Sample: {selected_sample}")
                    logger_ui.info(f"Loaded EIDO JSON data from sample: {selected_sample}")
            except FileNotFoundError:
                st.error(f"Sample file not found: {selected_sample}")
                logger_ui.error(f"Sample file not found: {selected_sample}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in sample {selected_sample}: {e}")
                logger_ui.error(f"Invalid JSON in sample {selected_sample}: {e}")
            except Exception as e:
                st.error(f"Error loading sample {selected_sample}: {e}")
                logger_ui.error(f"Failed loading sample {selected_sample}: {e}", exc_info=True)

        # 2. Gather Raw Alert Text
        if alert_text_input_area:
            alert_text_to_process = alert_text_input_area.strip()
            if alert_text_to_process:
                 logger_ui.debug("Gathered raw alert text from input area.")

        # 3. Check if any input was provided
        if not reports_to_process_from_sources and not alert_text_to_process:
            st.warning("No EIDO messages or alert text were provided.")
            logger_ui.warning("Processing button clicked, but no valid inputs found.")
        else:
            total_processed_count = 0
            total_error_count = 0
            total_start_time = time.time()
            status_messages = [] # Collect status messages

            # --- Process EIDO JSON First ---
            if reports_to_process_from_sources:
                final_reports_list = [] # Holds individual EIDO Message dicts
                error_count_loading = 0
                for idx, loaded_data in enumerate(reports_to_process_from_sources):
                    source_name = input_sources_names[idx] if idx < len(input_sources_names) else "Unknown Source"
                    if isinstance(loaded_data, list):
                        logger_ui.debug(f"Source '{source_name}' contained a list of {len(loaded_data)} items. Expanding.")
                        for item_index, item in enumerate(loaded_data):
                            if isinstance(item, dict):
                                final_reports_list.append(item)
                            else:
                                logger_ui.warning(f"Skipping non-dictionary item #{item_index+1} within list from source '{source_name}'. Type: {type(item)}")
                                error_count_loading += 1
                    elif isinstance(loaded_data, dict):
                        final_reports_list.append(loaded_data)
                    else:
                        logger_ui.error(f"Skipping invalid data type from source '{source_name}'. Expected dict or list, got {type(loaded_data)}.")
                        error_count_loading += 1

                num_reports_to_process = len(final_reports_list)
                if num_reports_to_process > 0:
                    logger_ui.info(f"Starting processing for {num_reports_to_process} individual EIDO message(s).")
                    st.info(f"Processing {num_reports_to_process} EIDO message(s)...")
                    progress_bar_json = st.progress(0)
                    processed_count_json = 0
                    error_count_json = 0

                    try:
                        for i, report_dict in enumerate(final_reports_list):
                            progress_val = (i + 1) / num_reports_to_process
                            progress_bar_json.progress(progress_val, text=f"Processing EIDO message {i+1}/{num_reports_to_process}")
                            msg_id_for_log = report_dict.get('eidoMessageIdentifier', report_dict.get('$id', f"unknown_json_{i+1}"))

                            try:
                                result_dict = eido_agent_instance.process_report_json(report_dict)
                                incident_id = result_dict.get('incident_id')
                                status_msg = result_dict.get('status', 'Unknown status from agent')
                                is_success = status_msg.lower() == "success"
                            except Exception as agent_call_error:
                                incident_id = None
                                is_success = False
                                status_msg = f"Agent processing failed critically: {agent_call_error}"
                                logger_ui.critical(f"CRITICAL ERROR calling agent for EIDO Msg ID '{msg_id_for_log}': {agent_call_error}", exc_info=True)

                            if is_success and incident_id:
                                processed_count_json += 1
                                logger_ui.info(f"EIDO Msg '{msg_id_for_log}': Successfully processed. Status: {status_msg}")
                            else:
                                error_count_json += 1
                                logger_ui.error(f"EIDO Msg '{msg_id_for_log}': Failed processing. Status: {status_msg}")
                                status_messages.append(f"EIDO Error (Msg: {msg_id_for_log[:15]}...): {status_msg}")

                        progress_bar_json.empty()
                        total_processed_count += processed_count_json
                        total_error_count += error_count_json
                        if error_count_json > 0:
                             status_messages.append(f"Processed {processed_count_json}/{num_reports_to_process} EIDO messages. {error_count_json} failed.")
                        elif error_count_loading > 0:
                             status_messages.append(f"Processed {processed_count_json}/{num_reports_to_process} EIDO messages. {error_count_loading} items skipped during loading.")
                        else:
                             status_messages.append(f"Processed {processed_count_json}/{num_reports_to_process} EIDO messages successfully.")

                    except Exception as agent_loop_error:
                         st.error(f"An unexpected error occurred during the EIDO processing loop: {agent_loop_error}")
                         logger_ui.critical(f"EIDO processing loop failed unexpectedly: {agent_loop_error}", exc_info=True)
                         if 'progress_bar_json' in locals() and progress_bar_json: progress_bar_json.empty()
                         total_error_count += num_reports_to_process - processed_count_json # Assume remaining failed
                         status_messages.append(f"EIDO processing loop failed: {agent_loop_error}")

            # --- Process Raw Alert Text Second ---
            if alert_text_to_process:
                 logger_ui.info("Starting processing for raw alert text.")
                 st.info("Processing raw alert text...")
                 progress_bar_text = st.progress(0, text="Parsing alert text...")
                 processed_count_text = 0
                 error_count_text = 0
                 try:
                     # Call the new agent method for text processing
                     result_dict = eido_agent_instance.process_alert_text(alert_text_to_process)
                     progress_bar_text.progress(1.0, text="Alert text processing complete.")

                     incident_id = result_dict.get('incident_id')
                     status_msg = result_dict.get('status', 'Unknown status from agent')
                     is_success = status_msg.lower() == "success"

                     if is_success and incident_id:
                         processed_count_text += 1
                         logger_ui.info(f"Raw Alert Text: Successfully processed. Status: {status_msg}")
                         status_messages.append(f"Raw Alert Text processed successfully.")
                     else:
                         error_count_text += 1
                         logger_ui.error(f"Raw Alert Text: Failed processing. Status: {status_msg}")
                         status_messages.append(f"Raw Alert Text Error: {status_msg}")

                 except Exception as agent_text_call_error:
                      error_count_text += 1
                      status_msg = f"Agent text processing failed critically: {agent_text_call_error}"
                      logger_ui.critical(f"CRITICAL ERROR calling agent for Raw Alert Text: {agent_text_call_error}", exc_info=True)
                      status_messages.append(f"Raw Alert Text Error: {status_msg}")

                 progress_bar_text.empty()
                 total_processed_count += processed_count_text
                 total_error_count += error_count_text

            # --- Display Overall Summary Message ---
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            if total_error_count > 0:
                 st.warning(f"Processing finished. {total_processed_count} item(s) succeeded, {total_error_count} failed. Took {total_duration:.2f}s.")
                 with st.expander("Show Processing Status Details"):
                     for msg in status_messages: st.write(msg)
            else:
                 st.success(f"Processing finished. {total_processed_count} item(s) succeeded. Took {total_duration:.2f}s.")
                 # Optionally show success messages too
                 # with st.expander("Show Processing Status Details"):
                 #    for msg in status_messages: st.write(msg)

            update_dashboard_data() # Refresh dashboard view and capture logs

        # --- Capture any final logs from processing ---
        get_captured_logs()


    # Admin Actions Expander
    st.markdown("---")
    with st.expander("⚙️ Admin Actions"):
        st.warning("These actions permanently modify the current session's data.")
        if st.button("Clear All Incidents", key="clear_button", type="secondary", use_container_width=True, help="Removes all incidents from the in-memory store for this session."):
             try:
                 count = len(incident_store.get_all_incidents())
                 if count > 0:
                      incident_store.clear_store()
                      logger_ui.warning(f"Admin action: Cleared {count} incidents from store.")
                      # Reset session state related to data
                      init_session_state() # Re-initialize to empty/default states
                      update_dashboard_data() # Update display (should show empty)
                      st.success(f"Cleared {count} incidents from the store.")
                      time.sleep(0.5)
                      st.rerun()
                 else:
                      st.info("Incident store is already empty.")
             except Exception as e:
                 st.error(f"Failed to clear incident store: {e}")
                 logger_ui.error(f"Error clearing incident store: {e}", exc_info=True)
             # Capture logs from clear action
             get_captured_logs()


    # Log Area Expander
    st.markdown("---")
    with st.expander("📄 Processing Log", expanded=False):
        # Ensure logs are updated before display
        get_captured_logs() # Call here to ensure latest logs are fetched
        if st.session_state.log_messages:
            log_display = "\n".join(st.session_state.log_messages) # Show newest first already handled by prepending
            st.code(log_display, language='log', line_numbers=False)
        else:
            st.caption("Log is empty.")

# --- Dashboard Area ---
with dashboard:
    st.header("📊 Incident Dashboard")
    # Key Metrics Row
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Incidents Tracked", st.session_state.total_incidents)
    metric_col2.metric("Currently Active Incidents", st.session_state.active_incidents)
    try:
        if not st.session_state.incidents_df.empty and 'Reports' in st.session_state.incidents_df.columns and st.session_state.incidents_df['Reports'].notna().any():
             avg_reports = st.session_state.incidents_df['Reports'].mean()
             metric_col3.metric("Avg. Reports / Incident", f"{avg_reports:.1f}")
        else:
             metric_col3.metric("Avg. Reports / Incident", "N/A")
    except Exception as e:
         logger_ui.warning(f"Error calculating avg reports metric: {e}", exc_info=True)
         metric_col3.metric("Avg. Reports / Incident", "Error")

    st.divider()

    # Tabs for different views
    tab_list, tab_map, tab_trends, tab_details = st.tabs([
        "🗓️ **Incident List**", "🗺️ **Geographic Map**", "📈 **Trends**", "🔍 **Details View**"
    ])

    # Incident List Tab
    with tab_list:
        st.subheader("Current Incident Overview")
        if not st.session_state.incidents_df.empty:
            df_display = st.session_state.incidents_df.copy()
            # Ensure 'Last Update' is datetime before sorting
            df_display['Last Update'] = pd.to_datetime(df_display['Last Update'], errors='coerce', utc=True)
            df_display.sort_values(by="Last Update", ascending=False, inplace=True, na_position='last')

            st.dataframe(
                df_display, use_container_width=True, hide_index=True,
                column_order=("ID", "Type", "Status", "Reports", "Last Update", "Actions", "Locations", "Summary"),
                column_config={
                    "ID": st.column_config.TextColumn("Incident ID", help="Short unique identifier", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="medium"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Reports": st.column_config.NumberColumn("Reports", format="%d", help="Number of associated reports processed for this incident", width="small"),
                    "Last Update": st.column_config.DatetimeColumn("Last Update (UTC)", format="YYYY-MM-DD HH:mm:ss", timezone="UTC"),
                    "Actions": st.column_config.TextColumn("Recommended Actions", width="large"),
                    "Locations": st.column_config.NumberColumn("Unique Locations", help="Number of unique geocoded locations", format="%d", width="small"),
                    "Summary": st.column_config.TextColumn("Latest Summary", width="medium"),
                }
            )
        else:
            st.info("No incidents processed yet. Use the sidebar to ingest EIDO messages or alert text.")

    # Geographic Map Tab
    with tab_map:
        st.subheader("Incident Locations")
        if not st.session_state.map_data.empty:
             try:
                 valid_map_data = st.session_state.map_data.dropna(subset=['lat', 'lon'])
                 if not valid_map_data.empty:
                     center_lat = valid_map_data['lat'].median()
                     center_lon = valid_map_data['lon'].median()
                     map_df_display = valid_map_data.copy()
                     map_df_display['size'] = 25
                     st.map(map_df_display, latitude='lat', longitude='lon', size='size', zoom=10)
                 else:
                      st.info("No valid geocoded location data available to display.")
             except Exception as map_error:
                 st.error(f"Error displaying map: {map_error}")
                 logger_ui.error(f"Map display error: {map_error}", exc_info=True)

             st.caption("Map displays unique geocoded locations from processed reports. Points may overlap.")
             with st.expander("Show Map Data Points"):
                 st.dataframe(
                     st.session_state.map_data[['incident_id', 'type', 'lat', 'lon']].round(6),
                     use_container_width=True, hide_index=True
                 )
        else:
            st.info("No valid geocoded location data available to display.")

    # Trends Tab
    with tab_trends:
        st.subheader("Incident Trends")
        if not st.session_state.incidents_df.empty:
            trends_col1, trends_col2 = st.columns(2)
            with trends_col1:
                st.markdown("##### Incident Types Distribution")
                if 'Type' in st.session_state.incidents_df.columns:
                    type_counts = st.session_state.incidents_df['Type'].value_counts()
                    if not type_counts.empty:
                        st.bar_chart(type_counts, use_container_width=True)
                    else:
                        st.caption("No incident type data available.")
                else:
                    st.warning("Incident 'Type' data missing in dashboard DataFrame.")

            with trends_col2:
                st.markdown("##### Report Activity Over Time")
                all_report_timestamps = []
                try:
                    for inc_obj in incident_store.get_all_incidents():
                        if inc_obj.reports_core_data:
                            for report_core in inc_obj.reports_core_data:
                                if report_core.timestamp:
                                    ts = pd.to_datetime(report_core.timestamp, errors='coerce', utc=True)
                                    if pd.notna(ts):
                                        all_report_timestamps.append(ts)
                                    else:
                                        logger_ui.warning(f"Skipping invalid report timestamp {report_core.timestamp} for trends (Report Core ID: {report_core.report_id[:8]}).")
                                else:
                                     logger_ui.warning(f"Skipping null report timestamp for trends (Report Core ID: {report_core.report_id[:8]}).")

                    if all_report_timestamps:
                        ts_series = pd.Series(1, index=pd.DatetimeIndex(all_report_timestamps)).sort_index()
                        freq_options = {'Hourly': 'h', 'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME'}
                        selected_freq_label = st.selectbox("Aggregate activity by:", options=list(freq_options.keys()), index=1, key="trend_freq_select")
                        freq_code = freq_options[selected_freq_label]
                        activity = ts_series.resample(freq_code).count()
                        if not activity.empty:
                            st.line_chart(activity, use_container_width=True)
                        else:
                            st.caption("No activity data for the selected period/frequency.")
                    else:
                        st.caption("No valid report timestamps found for activity trend.")
                except Exception as e:
                    st.error(f"Error generating activity trend: {e}")
                    logger_ui.error(f"Trend generation error: {e}", exc_info=True)

        else:
             st.info("Process incidents to view trends.")

    # Details View Tab - MODIFIED
    with tab_details:
        st.subheader("Detailed Incident Information")
        # Get current incidents directly from store for the selector
        current_incidents_map = {inc_obj.incident_id[:8]: inc_obj.incident_id for inc_obj in incident_store.get_all_incidents()}
        available_short_ids = sorted(list(current_incidents_map.keys()))

        if available_short_ids:
            selected_id_short = st.selectbox(
                "Select Incident ID:", options=["-- Select --"] + available_short_ids, index=0,
                key="detail_incident_selector", format_func=lambda x: f"Incident {x}" if x != "-- Select --" else x
            )

            if selected_id_short and selected_id_short != "-- Select --":
                full_incident_id = current_incidents_map.get(selected_id_short)
                selected_incident_obj = incident_store.get_incident(full_incident_id) if full_incident_id else None

                if selected_incident_obj:
                    st.markdown(f"#### Details for Incident `{selected_incident_obj.incident_id}`")

                    # --- Display Incident Details ---
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.markdown(f"**Type:** `{selected_incident_obj.incident_type or 'N/A'}`")
                        st.markdown(f"**Status:** `{selected_incident_obj.status or 'Unknown'}`")
                        report_count = selected_incident_obj.trend_data.get('report_count', 0)
                        st.markdown(f"**Associated Reports:** {report_count}")
                        loc_count = len(selected_incident_obj.locations)
                        st.markdown(f"**Unique Locations:** {loc_count}")
                    with detail_col2:
                         created_dt_str = pd.to_datetime(selected_incident_obj.created_at, errors='coerce', utc=True).strftime('%Y-%m-%d %H:%M:%S %Z') if pd.notna(selected_incident_obj.created_at) else "N/A"
                         updated_dt_str = pd.to_datetime(selected_incident_obj.last_updated_at, errors='coerce', utc=True).strftime('%Y-%m-%d %H:%M:%S %Z') if pd.notna(selected_incident_obj.last_updated_at) else "N/A"
                         st.markdown(f"**Created At (UTC):** {created_dt_str}")
                         st.markdown(f"**Last Updated (UTC):** {updated_dt_str}")
                         match_info = selected_incident_obj.trend_data.get('match_info', 'N/A')
                         st.markdown(f"**Last Match Info:** `{match_info}`")

                    st.divider()
                    st.markdown("##### 📄 Current Summary")
                    st.text_area(
                        "LLM Generated Summary:", value=selected_incident_obj.summary or "_No summary available._",
                        height=150, disabled=True, key=f"summary_{selected_id_short}"
                    )

                    st.markdown("##### ✅ Recommended Actions")
                    if selected_incident_obj.recommended_actions:
                         st.markdown("\n".join(f"- {action}" for action in selected_incident_obj.recommended_actions))
                    else:
                         st.markdown("_No specific actions recommended by the agent._")

                    st.divider()

                    # --- Associated Report Core Data Expander ---
                    reports_core_list = selected_incident_obj.reports_core_data or []
                    with st.expander(f"Show Associated Report Core Data ({len(reports_core_list)})", expanded=False):
                         if reports_core_list:
                              report_data_display = []
                              try:
                                   sorted_core_data = sorted(
                                       reports_core_list,
                                       key=lambda r: pd.to_datetime(r.timestamp, errors='coerce', utc=True) if r.timestamp else pd.Timestamp.min.replace(tzinfo=timezone.utc),
                                       reverse=True
                                   )
                              except Exception as sort_e:
                                   logger_ui.warning(f"Error sorting report core data for incident {selected_id_short}: {sort_e}", exc_info=True)
                                   sorted_core_data = reports_core_list

                              for i, r_core in enumerate(sorted_core_data):
                                  loc_display = "Unknown"
                                  addr = r_core.location_address
                                  coords_str = f"({r_core.coordinates[0]:.4f}, {r_core.coordinates[1]:.4f})" if r_core.coordinates and isinstance(r_core.coordinates, tuple) and len(r_core.coordinates)==2 else None
                                  if addr and coords_str: loc_display = f"{addr} {coords_str}"
                                  elif addr: loc_display = addr
                                  elif coords_str: loc_display = f"Coord: {coords_str}"
                                  ts_display = pd.to_datetime(r_core.timestamp, errors='coerce', utc=True)
                                  report_data_display.append({
                                      "#": i + 1, "Timestamp": ts_display,
                                      "Report Core ID": r_core.report_id[:8] if r_core.report_id else "N/A",
                                      "Orig EIDO Msg ID": r_core.original_document_id or "N/A",
                                      "Source Info": r_core.source or 'N/A',
                                      "Description/Notes": r_core.description or "N/A",
                                      "Location Info": loc_display
                                  })
                              st.dataframe(
                                    pd.DataFrame(report_data_display), use_container_width=True, hide_index=True,
                                    column_order=("#", "Timestamp", "Report Core ID", "Orig EIDO Msg ID", "Source Info", "Location Info", "Description/Notes"),
                                    column_config={
                                         "#": st.column_config.NumberColumn("Order", width="small", format="%d"),
                                         "Timestamp": st.column_config.DatetimeColumn("Timestamp (UTC)", format="YYYY-MM-DD HH:mm:ss", timezone="UTC"),
                                         "Report Core ID": st.column_config.TextColumn("Processed ID", width="small"),
                                         "Orig EIDO Msg ID": st.column_config.TextColumn("Original EIDO Msg ID", width="medium"),
                                         "Source Info": st.column_config.TextColumn("Source Info", width="medium"),
                                         "Location Info": st.column_config.TextColumn("Location Info", width="medium"),
                                         "Description/Notes": st.column_config.TextColumn("Description / Notes", width="large"),
                                    }
                                  )
                         else:
                              st.info("No report core data is associated with this incident record.")


                    # --- >>> NEW SECTION: Original EIDO JSON <<< ---
                    st.markdown("---") # Separator
                    st.markdown("##### 📜 Original EIDO Message(s) JSON")

                    original_eido_dicts = []
                    if reports_core_list:
                         # Sort reports by timestamp before extracting dicts to maintain order
                         try:
                              sorted_core_data_for_json = sorted(
                                  reports_core_list,
                                  key=lambda r: pd.to_datetime(r.timestamp, errors='coerce', utc=True) if r.timestamp else pd.Timestamp.min.replace(tzinfo=timezone.utc),
                                  reverse=False # Oldest first for list view
                              )
                         except Exception as sort_e:
                              logger_ui.warning(f"Error sorting report core data for JSON view: {sort_e}", exc_info=True)
                              sorted_core_data_for_json = reports_core_list # Fallback

                         for report_core in sorted_core_data_for_json:
                             if report_core.original_eido_dict and isinstance(report_core.original_eido_dict, dict):
                                 original_eido_dicts.append(report_core.original_eido_dict)

                    if original_eido_dicts:
                        # Prepare JSON string for display and download
                        # If only one message, display it as an object, otherwise as a list
                        display_data = original_eido_dicts[0] if len(original_eido_dicts) == 1 else original_eido_dicts
                        try:
                            json_string_all = json.dumps(display_data, indent=2)
                        except TypeError as json_err:
                             logger_ui.error(f"Failed to serialize original EIDO dicts for incident {selected_id_short}: {json_err}")
                             json_string_all = f'{{"error": "Could not serialize original EIDO data", "details": "{json_err}"}}'

                        st.text_area(
                            label="JSON Content (Copy from here):",
                            value=json_string_all,
                            height=300,
                            key=f"eido_json_display_{selected_id_short}",
                            help="Displays the original EIDO JSON message(s) associated with this incident. If multiple messages are associated, they are shown as a list.",
                            disabled=True # Make it read-only, user copies manually
                        )

                        # Download Button
                        st.download_button(
                            label="📥 Download All EIDO JSON",
                            data=json_string_all.encode('utf-8'), # Encode string to bytes
                            file_name=f"incident_{selected_id_short}_eido_messages.json",
                            mime="application/json",
                            key=f"download_eido_{selected_id_short}",
                            help="Download all associated original EIDO messages as a single JSON file (list if multiple)."
                        )
                    else:
                        st.info("No original EIDO JSON data found stored for this incident's reports.")
                    # --- >>> END NEW SECTION <<< ---

                else:
                    st.warning(f"Could not retrieve details for incident ID {selected_id_short}. It might have been cleared.")
            elif available_short_ids:
                 st.info("Select an Incident ID from the dropdown above to view its details.")
        else:
            st.info("Process EIDO messages or alert text via the sidebar to view incident details.")

# --- Footer ---
st.divider()
st.caption(f"EIDO Sentinel POC | v0.5.1 | Streamlit v{st.__version__} | Python {sys.version.split()[0]}") # Bump patch version

# Final log capture at the end of the run
get_captured_logs()