# EIDO Sentinel: AI Incident Processor

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%26%20FastAPI-ff69b4)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) <!-- Add a LICENSE file -->

🚨 Proof-of-Concept AI agent for processing emergency reports, primarily focusing on interpreting **Emergency Incident Data Object (EIDO)** messages.

This project demonstrates how AI, including Large Language Models (LLMs) accessed via providers like [Google Generative AI](https://ai.google.dev/) or [OpenRouter](https://openrouter.ai/), can process emergency reports. While initially designed around the NENA EIDO standard (NENA-STA-021.1a-2022 or later), the current implementation focuses on extracting core information from EIDO-like JSON structures to enable AI summarization and analysis, bypassing strict schema validation for flexibility during this POC stage.

The NENA EIDO standard defines a standardized JSON format for exchanging the *current state* of emergency incident information. More information can be found at the [NENA i3 Standards page](https://www.nena.org/page/i3_standards).

**Current Capabilities:**

*   **Ingest EIDO-like JSON Reports:** Parses JSON dictionaries representing emergency reports.
*   **Extract Core Data:** Identifies and extracts key fields like timestamps, incident types, descriptions, locations (including basic XML parsing for PIDF-LO), and source agencies using safe dictionary access.
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
        agent_core.py     # Main agent workflow
        llm_interface.py  # Interaction with LLMs
        matching.py       # Incident correlation logic
        __init__.py
    api/                  # FastAPI backend
        endpoints.py      # API route definitions
        main.py           # FastAPI application entry point
        __init__.py
    config/               # Configuration settings
        settings.py       # Environment variables and constants loading
        __init__.py
    data_models/          # Data structures and schemas
        eido_derived.schema.json # Original intended schema (currently bypassed for flexibility)
        schemas.py        # Pydantic models for INTERNAL objects (ReportCoreData, Incident)
        __init__.py
    sample_eido/          # Example EIDO JSON files for testing
        Additional samples.json
        Sample call transfer EIDO.json
        ucsd_alerts.json
    services/             # Supporting services (storage, geocoding, embeddings)
        embedding.py      # Text embedding generation
        geocoding.py      # Address to coordinates conversion
        storage.py        # In-memory incident storage
        __init__.py
    tests/                # Unit and integration tests (to be expanded)
        __init__.py
    ui/                   # Streamlit frontend application
        app.py            # Main Streamlit application script
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
*   An API Key for your chosen LLM provider (Google Generative AI or OpenRouter).

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
        *   `LLM_PROVIDER`: Set to `google`, `openrouter`, or `none`.
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

*   **Streamlit UI:** Use the sidebar to upload EIDO-like JSON files, paste JSON text, or load sample reports from the `sample_eido` directory. Click "Process Inputs" to run the agent. The dashboard tabs will update with incident lists, summaries, maps, basic trends, and recommended actions generated by the LLM. Use "Admin Actions" to clear stored incidents for the session. Check the "Processing Log" expander for detailed logs (set `LOG_LEVEL=DEBUG` in `.env` for maximum detail).
*   **FastAPI:** Use tools like `curl`, Postman, or Python `requests` to interact with the API endpoints (e.g., `POST /api/v1/ingest`, `GET /api/v1/incidents/{id}`). Explore endpoints via the `/docs` page.

## Next Steps / TODO

1.  **Agentic Alert-to-EIDO Conversion (PRIME NEXT STEP):**
    *   Develop a new agent component or workflow (`agent/alert_parser.py`?) responsible for taking various unstructured or semi-structured inputs (e.g., plain text UCSD alerts, police report summaries, transcribed 911 call snippets) and converting them into a valid NENA EIDO JSON structure.
    *   This will involve:
        *   Input parsing (text, potentially structured data).
        *   LLM-powered Named Entity Recognition (NER) and Relation Extraction to identify incident types, locations, times, people, vehicles, agencies, narrative details, etc.
        *   Mapping extracted information to the correct EIDO components and fields (including required NIEM elements and IANA registry codes where possible).
        *   Generating the final, schema-compliant EIDO JSON output.
        *   This generated EIDO can then be fed into the *existing* EIDO Sentinel processing pipeline.

2.  **Improve EIDO Processing Compliance & Depth:**
    *   *(See detailed section below)* Reintroduce stricter schema validation, fully parse NIEM components, leverage IANA codes, refine "current state" logic, follow `$ref` links, handle advanced location types, and interpret split/merge/link components.

3.  **Enhance Core Functionality:**
    *   Implement vector-based similarity search (`agent/matching.py`, `services/storage.py`) using embeddings for more robust incident correlation.
    *   Add comprehensive unit and integration tests (`tests/`).
    *   Refine LLM prompts for edge cases and potentially structured output (JSON).
    *   Implement persistent storage (`services/storage.py` - e.g., SQLite, PostgreSQL).
    *   Improve error handling for external services and invalid inputs.
    *   Add user authentication/authorization if needed.

## Contributing

Contributions are welcome! Please follow standard Git workflow (fork, branch, pull request). Ensure code follows basic formatting and includes docstrings. Please update tests for any new functionality.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Detailed Next Steps to Improve EIDO Handling

*(These steps focus on making the existing EIDO processing more robust and compliant with the NENA standard, complementing the primary goal of adding Alert-to-EIDO conversion.)*

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