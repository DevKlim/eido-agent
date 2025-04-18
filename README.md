# EIDO Sentinel: AI Incident Processor

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%26%20FastAPI-ff69b4)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) <!-- Add a LICENSE file -->

🚨 Proof-of-Concept AI agent for processing emergency reports, focusing on interpreting **Emergency Incident Data Object (EIDO)** messages and converting unstructured alerts into EIDO-like structures.

This project demonstrates how AI, including Large Language Models (LLMs) accessed via providers like [Google Generative AI](https://ai.google.dev/) or [OpenRouter](https://openrouter.ai/), can process emergency reports. It can:
1.  Ingest and process standardized NENA EIDO messages (NENA-STA-021.1a-2022 or later), focusing on extracting core information from the JSON structure while bypassing strict schema validation for flexibility during this POC stage.
2.  Ingest **unstructured alert text** (e.g., CAD summaries, SMS alerts, call transcript snippets) and use an LLM to parse it into an EIDO-like JSON dictionary, which is then processed by the same pipeline.

The NENA EIDO standard defines a standardized JSON format for exchanging the *current state* of emergency incident information. More information can be found at the [NENA i3 Standards page](https://www.nena.org/page/i3_standards).

**Current Capabilities:**

*   **Ingest EIDO JSON Reports:** Parses JSON dictionaries representing EIDO messages.
*   **Ingest Raw Alert Text:** Parses unstructured text using an LLM (`agent/alert_parser.py`) to generate an EIDO-like JSON dictionary.
*   **Extract Core Data:** Identifies and extracts key fields like timestamps, incident types, descriptions, locations (including basic XML parsing for PIDF-LO), and source agencies using safe dictionary access from EIDO (or EIDO-like) dictionaries.
*   **Store Original Data:** Retains the original input EIDO JSON dictionary (or the AI-generated one from raw text) associated with each processed report.
*   **Geocode Locations:** Converts extracted addresses to geographic coordinates using Nominatim (requires configuration).
*   **Correlate Incidents (Basic):** Determines if an incoming report represents a new incident or an update based on time windows, location proximity, and external IDs.
*   **Generate Incident Summaries & Actions:** Uses a configured LLM (Google Gemini or via OpenRouter) to generate evolving incident summaries and suggest recommended actions based on the extracted core data and history.
*   **In-Memory Storage:** Stores and manages consolidated incident data during the application's runtime.
*   **Provide Interfaces:** Offers an interactive dashboard (Streamlit) for visualization and testing, and basic API endpoints (FastAPI) for potential integration.

## Project Structure

The project is organized into logical modules:
```
./
    .env.example
    .gitignore
    install_dependencies.sh
    print_struc.py
    README.md
    requirements.txt
    run_api.sh
    run_streamlit.sh
    agent/                # Core AI agent logic
        agent_core.py     # Main agent workflow (handles both JSON and text)
        alert_parser.py   # Logic for parsing raw text to EIDO-like dict using LLM
        llm_interface.py  # Interaction with LLMs
        matching.py       # Incident correlation logic
        __init__.py
    api/                  # FastAPI backend
        endpoints.py      # API route definitions (includes endpoint for raw text)
        main.py           # FastAPI application entry point
        __init__.py
    config/               # Configuration settings
        settings.py       # Environment variables and constants loading
        __init__.py
    data_models/          # Data structures and schemas
        eido_derived.schema.json # Original intended schema (currently bypassed for flexibility)
        schemas.py        # Pydantic models for INTERNAL objects (ReportCoreData includes original_eido_dict, Incident)
        __init__.py
    sample_eido/          # Example EIDO JSON files and potentially raw text samples
        Additional samples.json
        Sample call transfer EIDO.json
        ucsd_alerts.json
        # sample_alert.txt (Example raw text file could be added)
    services/             # Supporting services (storage, geocoding, embeddings)
        embedding.py      # Text embedding generation
        geocoding.py      # Address to coordinates conversion
        storage.py        # In-memory incident storage
        __init__.py
    tests/                # Unit and integration tests (to be expanded)
        __init__.py
    ui/                   # Streamlit frontend application
        app.py            # Main Streamlit application script (includes raw text input tab, JSON download)
        components.py     # Reusable UI components (if any)
        __init__.py
    utils/                # Helper functions and utilities
        helpers.py        # General utility functions (if any)
        __init__.py
```

## Getting Started

### Prerequisites

*   Python 3.9+
*   Pip (Python package installer)
*   Git
*   (Optional but Recommended) A virtual environment tool (`venv`)
*   An API Key for your chosen LLM provider (Google Generative AI or OpenRouter). Required for summarization, recommendations, and **raw text parsing**.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd eido-sentinel # Or your chosen directory name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    *   **On Linux/macOS:** Use the provided script:
        ```bash
        chmod +x install_dependencies.sh
        ./install_dependencies.sh
        ```
    *   **On Windows (or manually):**
        ```bash
        pip install -r requirements.txt
        ```

4.  **Set up environment variables (CRITICAL STEP):**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   **Edit the `.env` file** with your specific settings:
        ```bash
        # Example using nano editor:
        nano .env
        # Or use vim, VS Code, Notepad, etc.
        ```
    *   **REQUIRED Settings:**
        *   `LLM_PROVIDER`: Set to `google`, `openrouter`, or `none`. **Must be `google` or `openrouter` for raw text parsing.**
        *   `GEOCODING_USER_AGENT`: Set a descriptive user agent for Nominatim (OpenStreetMap) geocoding as per their policy (e.g., `"MyEidoApp/1.0 (myemail@domain.com)"`). **Using a valid contact is essential.**
    *   **Conditional Settings (based on `LLM_PROVIDER`):**
        *   If `LLM_PROVIDER=google`:
            *   `GOOGLE_API_KEY`: Your Google Generative AI API key.
            *   `GOOGLE_MODEL_NAME`: (Optional, defaults provided) e.g., `"gemini-1.5-flash-latest"`
        *   If `LLM_PROVIDER=openrouter`:
            *   `LLM_API_KEY`: Your **OpenRouter API key** (starts with `sk-or-`).
            *   `LLM_MODEL_NAME`: The **OpenRouter model identifier** (e.g., `"openai/gpt-4o-mini"`, `"anthropic/claude-3-haiku-20240307"`). See [OpenRouter Docs](https://openrouter.ai/docs#models).
            *   `LLM_API_BASE_URL`: Should be `"https://openrouter.ai/api/v1"`.
    *   **Optional Settings:**
        *   `LOG_LEVEL`: Set logging verbosity (e.g., `DEBUG`, `INFO`). Defaults to `INFO`.
        *   `EMBEDDING_MODEL_NAME`: Change the default SentenceTransformer model.
        *   `SIMILARITY_THRESHOLD`, `TIME_WINDOW_MINUTES`, `DISTANCE_THRESHOLD_KM`: Adjust incident matching parameters.
        *   `ALERT_PARSING_PROMPT_PATH`: (Optional) Path to a custom prompt file for the alert-to-EIDO parser.

### Running the Application

Use the provided shell scripts (ensure they are executable: `chmod +x *.sh`). They will attempt to activate the `venv` automatically.

*   **Run the Streamlit UI:**
    ```bash
    ./run_streamlit.sh
    ```
    Access the dashboard in your browser (usually at `http://localhost:8501` or similar).

*   **Run the FastAPI API:**
    ```bash
    ./run_api.sh
    ```
    The API will be available (usually at `http://localhost:8000`). Access the interactive Swagger documentation at `http://localhost:8000/docs`.

*(Note: On Windows, run the `streamlit run ui/app.py` and `uvicorn api.main:app --reload` commands manually after activating the virtual environment).*

## Usage

*   **Streamlit UI:**
    *   **Input:** Use the sidebar tabs:
        *   "EIDO JSON Input": Upload EIDO JSON files, paste JSON text, or load sample reports from the `sample_eido` directory.
        *   "Raw Alert Text Input": Paste unstructured alert text (requires LLM configured).
    *   **Processing:** Click "Process Inputs" to run the agent on the provided data.
    *   **Dashboard:** The main area tabs will update:
        *   "Incident List": Shows tracked incidents.
        *   "Geographic Map": Displays geocoded locations.
        *   "Trends": Basic charts on incident types and activity.
        *   "Details View": Select an incident ID to see its summary, recommended actions, associated report core data, and **view/copy/download the original EIDO JSON** (or the AI-generated JSON from raw text).
    *   **Admin:** Use "Admin Actions" to clear stored incidents for the session. Check the "Processing Log" expander for detailed logs (set `LOG_LEVEL=DEBUG` in `.env` for maximum detail).
*   **FastAPI:** Use tools like `curl`, Postman, or Python `requests` to interact with the API endpoints (e.g., `POST /api/v1/ingest` for EIDO JSON, `POST /api/v1/ingest_alert` for raw text, `GET /api/v1/incidents/{id}`). Explore endpoints via the `/docs` page.

## Next Steps / TODO

1.  **Enhanced Agent Control & Visualization Dashboard (PRIME NEXT STEP):**
    *   **Goal:** Develop a more sophisticated UI/dashboard focused on the agentic parsing workflow and overall incident management.
    *   **Multi-Format Input:** Extend the agent (`agent/alert_parser.py`, potentially new modules) and UI (`ui/app.py`) to handle diverse input formats beyond plain text (e.g., structured logs, potentially PDF reports, email bodies).
    *   **Parsing Visualization:** In the UI, visualize the LLM's parsing process for unstructured inputs. Show the original text, highlight extracted entities (incident type, location, time, etc.), display confidence scores (if available from the LLM/parsing logic), and show the resulting generated EIDO-like JSON side-by-side.
    *   **EIDO Visualization:** Enhance the display of EIDO data (original or generated). Instead of just raw JSON, provide a structured, human-readable view of key EIDO components and fields.
    *   **Human-in-the-Loop Correction:** Allow users to review the AI's parsing results (extracted entities, generated EIDO fields) in the UI and make corrections *before* the data is finalized and processed. This corrected data could potentially be used for fine-tuning the parsing prompts/logic later.
    *   **Agent Configuration/Prompt Management:** Add a UI section (potentially admin-only) to view, manage, and possibly test different prompts used for alert parsing, summarization, and action recommendation.
    *   **Improved Incident Visualization:** Enhance existing dashboard tabs (map with clustering/heatmaps, more insightful trends, incident relationship graphs if split/merge implemented).

2.  **Improve EIDO Processing Compliance & Depth:**
    *   *(See detailed section below)* Reintroduce stricter schema validation, fully parse NIEM components, leverage IANA codes, refine "current state" logic, follow `$ref` links, handle advanced location types, and interpret split/merge/link components.

3.  **Enhance Core Functionality:**
    *   Implement vector-based similarity search (`agent/matching.py`, `services/storage.py`) using embeddings for more robust incident correlation, especially for text-based inputs.
    *   Add comprehensive unit and integration tests (`tests/`), particularly for the parsing and matching logic.
    *   Refine LLM prompts (parsing, summarization, actions) for edge cases, improved accuracy, and potentially structured output (JSON mode).
    *   Implement persistent storage (`services/storage.py` - e.g., SQLite, PostgreSQL) instead of in-memory only.
    *   Improve error handling for external services (LLM, Geocoding) and invalid inputs.
    *   Add user authentication/authorization if needed for production use.

## Contributing

Contributions are welcome! Please follow standard Git workflow (fork, branch, pull request). Ensure code follows basic formatting and includes docstrings. Please update tests for any new functionality.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Detailed Next Steps to Improve EIDO Handling

*(These steps focus on making the existing EIDO processing more robust and compliant with the NENA standard, complementing the primary goal of enhancing the agentic parsing dashboard.)*

1.  **Full Schema Validation & Handling:**
    *   Refine Pydantic models (`data_models/schemas.py`) to accurately reflect the official NENA EIDO OpenAPI schema.
    *   Implement strict validation during ingestion (`agent/agent_core.py`, `api/endpoints.py`).
    *   Gracefully handle optional/missing components.

2.  **Deep NIEM Component Integration:**
    *   Implement full parsing of embedded NIEM types (`nc:PersonType`, `nc:VehicleType`, `nc:LocationType`, etc.).
    *   Extract specific NIEM fields to enrich agent understanding. Map vCard data.

3.  **Leverage IANA Registries:**
    *   Integrate knowledge of EIDO-specific IANA registry codes (`incidentTypeCommonRegistryText`, `incidentStatusCommonRegistryText`, `dispositionCommonRegistryCode`, etc.) into agent logic and LLM prompts.

4.  **Refine "Current State" Processing:**
    *   Enhance update handling (`agent/agent_core.py`) to identify *changes* between sequential EIDOs for an incident.
    *   Instruct the LLM to summarize based on the *latest state* and identified changes.
    *   Ensure internal storage (`services/storage.py`) reflects the latest known state.

5.  **Exploit Component Relationships (`$ref`):**
    *   Correctly parse and utilize `$ref` links between components to build a connected incident graph.

6.  **Advanced Location Handling:**
    *   Robustly parse PIDF-LO (value/reference) and potential NIEM location formats.
    *   Handle cross streets and use `locationTypeDescriptionRegistryText`.

7.  **Interpret Split/Merge/Link Components:**
    *   Add logic (`agent/agent_core.py`, `agent/matching.py`) to recognize and act upon `MergeComponent` and `LinkComponent` data.

8.  **Contextual LLM Prompting with EIDO Specifics:**
    *   Refine LLM prompts (`agent/llm_interface.py`) to include key structured EIDO data (priority, type codes, status codes, NIEM attributes) for more grounded reasoning.
