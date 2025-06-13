#!/bin/bash

# EIDO Sentinel - run_all.sh
# Enhanced for robustness and better error reporting.

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
ENV_FILE="${SCRIPT_DIR}/.env"

DEFAULT_API_PORT="8000"
DEFAULT_STREAMLIT_PORT="8501"
FASTAPI_PARENT_PID="" # Stores PID of the uvicorn --reload parent process

# --- Temporary Directory for Script Operations ---
# Needs to be created early for logs, cleaned up by trap
TMP_DIR=$(mktemp -d -t eido_sentinel_run_XXXXXX)
if [ ! -d "$TMP_DIR" ]; then
    echo "FATAL: Could not create temporary directory. Exiting."
    exit 1
fi

# --- Helper Functions ---
log_info() {
    echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1" >&2
}

# --- Function to kill processes on specific ports ---
# Takes port numbers as arguments
kill_processes_on_ports() {
    if [ $# -eq 0 ]; then
        log_warn "kill_processes_on_ports: No ports specified."
        return
    fi

    for port_to_kill in "$@"; do
        if [ -z "$port_to_kill" ]; then
            log_warn "kill_processes_on_ports: Empty port number received. Skipping."
            continue
        fi

        log_info "Attempting to free port ${port_to_kill}..."
        local pids_found=() # Array to store PIDs

        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
            # Windows (Git Bash, Cygwin, etc.)
            log_info "OS Type: Windows. Using netstat and taskkill for port ${port_to_kill}."
            
            local netstat_output_file="${TMP_DIR}/netstat_port_${port_to_kill}.log"
            netstat -aon > "$netstat_output_file"
            if [ $? -ne 0 ]; then
                log_error "netstat command failed. Cannot check port ${port_to_kill}."
                rm -f "$netstat_output_file"
                continue
            fi

            # Extract PIDs listening on the specific port
            while IFS= read -r pid_val; do
                clean_pid=$(echo "$pid_val" | tr -d '\r\n[:space:]')
                if [[ "$clean_pid" =~ ^[0-9]+$ && "$clean_pid" != "0" ]]; then
                    pids_found+=("$clean_pid")
                fi
            done < <(grep ":${port_to_kill}[[:space:]]" "$netstat_output_file" | awk '$4 == "LISTENING" {print $NF}' | sort -u)
            
            rm -f "$netstat_output_file"

            if [ ${#pids_found[@]} -gt 0 ]; then
                log_info "Found PIDs on port ${port_to_kill}: ${pids_found[*]}"
                for pid_k in "${pids_found[@]}"; do
                    log_info "Attempting to terminate process (PID: $pid_k) on port ${port_to_kill}..."
                    local taskkill_log="${TMP_DIR}/taskkill_${pid_k}.log"
                    
                    cmd.exe /c "taskkill /F /PID $pid_k" > "$taskkill_log" 2>&1
                    local taskkill_exit_code=$?
                    
                    if [ $taskkill_exit_code -eq 0 ]; then
                        log_info "SUCCESS: taskkill for PID $pid_k completed."
                    elif [ $taskkill_exit_code -eq 128 ]; then
                        log_info "INFO: taskkill reported PID $pid_k not found (Error 128). Likely already terminated."
                    else
                        log_error "taskkill for PID $pid_k failed with exit code $taskkill_exit_code."
                        if [ -s "$taskkill_log" ]; then
                           log_error "Taskkill output for PID $pid_k:"; cat "$taskkill_log";
                        fi
                    fi
                    rm -f "$taskkill_log"
                done
                log_info "Waiting a moment for port ${port_to_kill} to free up after taskkill attempts..."
                sleep 3
            else
                log_info "No listening process found on port ${port_to_kill} by netstat."
            fi
        else
            # Linux, macOS
            log_info "OS Type: Non-Windows. Using lsof for port ${port_to_kill}."
            if command -v lsof > /dev/null; then
                local pids_lsof
                pids_lsof=$(lsof -t -i:"${port_to_kill}" -sTCP:LISTEN)
                
                if [ -n "$pids_lsof" ]; then
                    log_info "Found PIDs on port ${port_to_kill} via lsof: $pids_lsof. Terminating..."
                    for single_pid in $pids_lsof; do kill -9 "$single_pid"; done
                    log_info "Waiting a moment for port ${port_to_kill} to free up..."; sleep 1
                else
                    log_info "No listening process found on port ${port_to_kill} by lsof."
                fi
            else
                 log_warn "'lsof' command not found. Cannot check/kill processes on port ${port_to_kill}."
            fi
        fi
        log_info "Finished attempt to free port ${port_to_kill}."; echo "---"
    done
}

# --- Function to clean up background processes on script exit ---
cleanup() {
    log_info "Performing cleanup..."
    
    if [ -n "$FASTAPI_PARENT_PID" ]; then
        if kill -0 "$FASTAPI_PARENT_PID" > /dev/null 2>&1; then
            log_info "Stopping FastAPI parent process (PID: $FASTAPI_PARENT_PID)..."
            kill "$FASTAPI_PARENT_PID" # Send SIGTERM
            sleep 1
            if kill -0 "$FASTAPI_PARENT_PID" > /dev/null 2>&1; then
                log_warn "FastAPI parent process (PID: $FASTAPI_PARENT_PID) still running. Forcing kill (SIGKILL)..."
                kill -9 "$FASTAPI_PARENT_PID"
            fi
        fi
        FASTAPI_PARENT_PID=""
    fi
    
    log_info "Removing temporary directory: ${TMP_DIR}"; rm -rf "${TMP_DIR}"
    log_info "Cleanup complete."
}

# Trap EXIT, SIGINT (Ctrl+C) and SIGTERM to run cleanup
trap cleanup EXIT SIGINT SIGTERM

# --- Main Script ---
log_info "Launching EIDO Sentinel (FastAPI Backend & Streamlit UI)..."
cd "$SCRIPT_DIR"

# --- Activate Virtual Environment ---
if [ -d "${VENV_DIR}" ]; then
    log_info "Activating virtual environment: ${VENV_DIR}"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        source "${VENV_DIR}/Scripts/activate"
    else
        source "${VENV_DIR}/bin/activate"
    fi
    if [ $? -ne 0 ]; then
        log_error "Failed to activate virtual environment. Please check venv setup."; exit 1;
    fi
else
    log_warn "Virtual environment '${VENV_DIR}' not found. Using system Python.";
fi

# --- Check for dependencies ---
for cmd in uvicorn streamlit; do
    if ! command -v $cmd &> /dev/null; then
        log_error "'$cmd' command not found. Please ensure dependencies are installed (e.g., pip install -r requirements.txt)."; exit 1;
    fi
done

# --- Load variables from .env file ---
if [ -f "${ENV_FILE}" ]; then
    log_info "Loading environment variables from ${ENV_FILE}..."; set -o allexport; source "${ENV_FILE}"; set +o allexport;
else
    log_info "${ENV_FILE} not found. Using default port configurations."
fi

API_PORT_TO_USE="${API_PORT:-$DEFAULT_API_PORT}"
STREAMLIT_PORT_TO_USE="${STREAMLIT_SERVER_PORT:-$DEFAULT_STREAMLIT_PORT}"
API_HOST_TO_USE="${API_HOST:-127.0.0.1}"

log_info "Effective API Host: $API_HOST_TO_USE"; log_info "Effective API Port: $API_PORT_TO_USE"; log_info "Effective Streamlit Port: $STREAMLIT_PORT_TO_USE"

# --- Kill any pre-existing processes on the ports ---
log_info "Performing initial port cleanup..."; kill_processes_on_ports "$API_PORT_TO_USE" "$STREAMLIT_PORT_TO_USE"; log_info "Initial port cleanup finished."

# --- Start FastAPI Backend with robust logging ---
log_info "Starting FastAPI backend with uvicorn on ${API_HOST_TO_USE}:${API_PORT_TO_USE}..."
UVICORN_STDOUT_LOG="${TMP_DIR}/uvicorn_stdout.log"
UVICORN_STDERR_LOG="${TMP_DIR}/uvicorn_stderr.log"
log_info "Backend logs will be saved in ${TMP_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
uvicorn api.main:app --reload --log-level info --host "${API_HOST_TO_USE}" --port "${API_PORT_TO_USE}" > "$UVICORN_STDOUT_LOG" 2> "$UVICORN_STDERR_LOG" &
FASTAPI_PARENT_PID=$!
log_info "FastAPI backend initiated (Uvicorn Parent PID: $FASTAPI_PARENT_PID)."
log_info "Access API docs at http://${API_HOST_TO_USE}:${API_PORT_TO_USE}/docs"
log_info "Waiting for FastAPI to initialize (5 seconds)..."; sleep 5

# Check if FastAPI parent process started correctly and display logs on failure
if ! kill -0 "$FASTAPI_PARENT_PID" > /dev/null 2>&1; then
    log_error "FATAL: FastAPI backend parent process (PID: $FASTAPI_PARENT_PID) failed to start or exited prematurely."
    log_error "This is often due to a configuration error (e.g., database connection) or a missing dependency."
    if [ -s "$UVICORN_STDERR_LOG" ]; then
        log_error "Displaying backend error log from: $UVICORN_STDERR_LOG"
        echo "--- BACKEND ERROR LOG START ---"
        cat "$UVICORN_STDERR_LOG"
        echo "--- BACKEND ERROR LOG END ---"
    else
        log_warn "Backend error log is empty, checking stdout..."
        if [ -s "$UVICORN_STDOUT_LOG" ]; then
            log_error "Displaying backend standard output log from: $UVICORN_STDOUT_LOG"
            echo "--- BACKEND STDOUT LOG START ---"
            cat "$UVICORN_STDOUT_LOG"
            echo "--- BACKEND STDOUT LOG END ---"
        fi
    fi
    log_error "Exiting now. Please fix the backend issue and restart."
    exit 1
fi
log_info "FastAPI parent process seems to be running."

# --- Start Streamlit UI in the foreground ---
log_info "Attempting to start Streamlit UI on port ${STREAMLIT_PORT_TO_USE}..."
log_info "Streamlit will run in the foreground. Press Ctrl+C to stop both services."

streamlit run ui/app.py --server.port "${STREAMLIT_PORT_TO_USE}" --server.headless true

STREAMLIT_EXIT_CODE=$? 
log_info "Streamlit command has exited with code: $STREAMLIT_EXIT_CODE."

exit $STREAMLIT_EXIT_CODE