#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_CMD="python3" # Use python3 explicitly

# --- Colors for Output ---
COLOR_RESET='\033[0m'
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[0;33m'
COLOR_BLUE='\033[0;34m'

# --- Helper Functions ---
print_info() {
    echo -e "${COLOR_BLUE}INFO:${COLOR_RESET} $1"
}

print_success() {
    echo -e "${COLOR_GREEN}SUCCESS:${COLOR_RESET} $1"
}

print_warning() {
    echo -e "${COLOR_YELLOW}WARNING:${COLOR_RESET} $1"
}

print_error() {
    echo -e "${COLOR_RED}ERROR:${COLOR_RESET} $1" >&2 # Print errors to stderr
}

# --- Main Script Logic ---

print_info "Starting EIDO Sentinel Dependency Installation..."
echo "--------------------------------------------------"

# 1. Check for Python 3
print_info "Checking for ${PYTHON_CMD}..."
if ! command -v ${PYTHON_CMD} &> /dev/null; then
    print_error "${PYTHON_CMD} command not found. Please install Python 3 (>= 3.9 recommended)."
    exit 1
fi
PYTHON_VERSION=$(${PYTHON_CMD} --version)
print_success "Found ${PYTHON_CMD}: ${PYTHON_VERSION}"

# 2. Check for pip
print_info "Checking for pip..."
# Ensure pip is available for the selected python command
if ! ${PYTHON_CMD} -m pip --version &> /dev/null; then
    print_error "pip module not found for ${PYTHON_CMD}. Please ensure pip is installed."
    print_error "Try running: sudo apt update && sudo apt install ${PYTHON_CMD}-pip (Debian/Ubuntu)"
    print_error "Or: brew install python3 (macOS with Homebrew)"
    exit 1
fi
PIP_VERSION=$(${PYTHON_CMD} -m pip --version)
print_success "Found pip: ${PIP_VERSION}"


# 3. Check for requirements.txt
print_info "Checking for ${REQUIREMENTS_FILE}..."
if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    print_error "${REQUIREMENTS_FILE} not found in the current directory ($(pwd))."
    print_error "Please ensure you are running this script from the project root directory."
    exit 1
fi
print_success "${REQUIREMENTS_FILE} found."

# 4. Create Virtual Environment if it doesn't exist
print_info "Setting up virtual environment in './${VENV_DIR}'..."
if [ ! -d "${VENV_DIR}" ]; then
    ${PYTHON_CMD} -m venv "${VENV_DIR}"
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment."
        exit 1
    fi
    print_success "Virtual environment created."
else
    print_info "Virtual environment './${VENV_DIR}' already exists. Skipping creation."
fi

# Determine paths for venv executables based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash, Cygwin, etc.)
    VENV_PYTHON="${VENV_DIR}/Scripts/python"
    VENV_PIP="${VENV_DIR}/Scripts/pip"
    ACTIVATE_CMD="source ${VENV_DIR}/Scripts/activate"
else
    # Linux, macOS, etc.
    VENV_PYTHON="${VENV_DIR}/bin/python"
    VENV_PIP="${VENV_DIR}/bin/pip"
    ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
fi


# Ensure venv pip exists
if [ ! -f "${VENV_PIP}" ]; then
   print_error "Cannot find pip executable in virtual environment: ${VENV_PIP}"
   print_error "The virtual environment might be corrupted. Try removing the '${VENV_DIR}' directory and running the script again."
   exit 1
fi

# 5. Install dependencies using the virtual environment's pip
print_info "Installing dependencies from ${REQUIREMENTS_FILE} into the virtual environment..."
"${VENV_PIP}" install -r "${REQUIREMENTS_FILE}"
if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies from ${REQUIREMENTS_FILE}."
    print_error "Check the output above for specific package installation errors."
    exit 1
fi
print_success "Dependencies installed successfully!"

# 6. Reminder for .env file
echo "--------------------------------------------------"
print_info "Reminder: Ensure your environment variables are set."
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    print_warning "'.env' file not found. You may need to copy '.env.example' to '.env' and configure it:"
    echo "  cp .env.example .env"
    echo "  # Then edit .env with your settings (e.g., API keys, geocoding user agent)"
elif [ -f ".env" ]; then
     print_success "'.env' file found. Ensure it contains the necessary settings."
fi

# 7. Final Instructions
echo "--------------------------------------------------"
print_success "Installation Complete!"
echo "--------------------------------------------------"
echo ""
echo "To activate the virtual environment, run the following command in your terminal:"
echo -e "  ${COLOR_YELLOW}${ACTIVATE_CMD}${COLOR_RESET}"
echo ""
echo "Once the environment is activated (you should see '(${VENV_DIR})' in your prompt),"
echo "you can run the applications:"
echo ""
echo "  To start the Streamlit UI:"
echo -e "    ${COLOR_GREEN}streamlit run ui/app.py${COLOR_RESET}"
echo ""
echo "  To start the FastAPI API (in another terminal):"
echo -e "    ${COLOR_GREEN}uvicorn api.main:app --reload --port 8000${COLOR_RESET}"
echo ""
echo "To deactivate the virtual environment when finished, simply run:"
echo -e "  ${COLOR_YELLOW}deactivate${COLOR_RESET}"
echo ""

exit 0