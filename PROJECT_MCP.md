# EIDO Sentinel: Master Control Program & File Explanations

## Project Overview

**EIDO Sentinel** is an AI-powered platform designed to enhance emergency response by intelligently processing, correlating, and analyzing diverse emergency data streams. It leverages NENA EIDO standards, LLMs, and agentic AI principles to transform data into actionable insights.

**Version:** 0.9.1 (as of this document)

## Table of Contents

1.  [Root Directory Files](#1-root-directory-files)
2.  [Agent Module (`agent/`)](#2-agent-module-agent)
3.  [API Module (`api/`)](#3-api-module-api)
4.  [Configuration Module (`config/`)](#4-configuration-module-config)
5.  [Data Models Module (`data_models/`)](#5-data-models-module-data_models)
6.  [EIDO Templates (`eido_templates/`)](#6-eido-templates-eido_templates)
7.  [Services Module (`services/`)](#7-services-module-services)
8.  [Static Assets (`static/`)](#8-static-assets-static)
9.  [User Interface Module (`ui/`)](#9-user-interface-module-ui)
10. [Utilities Module (`utils/`)](#10-utilities-module-utils)

---

## 1. Root Directory Files

### 1.1. `.env.example`
*   **Purpose:** Serves as a template for required environment variables for **local development**. Users must copy this to `.env` and fill in values. For deployment, these variables are set in the hosting provider's interface.
*   **Key Aspects/Logic:**
    *   **`API_BASE_URL`**: **Crucial for deployment.** Locally, it points to the local backend. When deployed, the frontend (Streamlit Cloud) must have this set to the public URL of the deployed backend (Render).
    *   **Backend variables**: `DATABASE_URL`, `LLM_PROVIDER`, API keys, etc., are configured in the backend's hosting environment.
    *   **Deployment Notes**: Contains explicit instructions on how to configure secrets for both backend (Render) and frontend (Streamlit Cloud) environments.

### 1.2. `.gitignore`
*   **Purpose:** Specifies untracked files for Git to ignore, keeping the repository clean of caches, virtual environments, and sensitive files like `.env`.

### 1.3. `llm_project_context.md`
*   **Purpose:** A detailed project context document intended for an LLM to understand the project's mission, architecture, and features, aiding in AI-assisted development.

### 1.4. `run_all.sh`
*   **Purpose:** A local development script to start both the FastAPI backend and Streamlit UI, sourcing environment variables from `.env`.

---

## 2. Agent Module (`agent/`)

The `agent/` module contains the core intelligence of EIDO Sentinel.

### 2.1. `agent/__init__.py`
*   **Purpose:** An empty file that marks the `agent` directory as a Python package. This allows modules within `agent` to be imported using package notation (e.g., `from agent.agent_core import EidoAgent`).

### 2.2. `agent/agent_core.py`
*   **Purpose:** This is the heart of the EIDO Sentinel's processing logic. The `EidoAgent` class orchestrates the entire pipeline from data ingestion to incident creation/update and LLM-driven analysis, including automated geocoding.
*   **Key Class: `EidoAgent`**
    *   **`__init__(self)`**: Initializes the agent, primarily setting up an `IncidentStore` instance for database interaction.
    *   **`_resolve_ref_string_from_dict(self, ref_input)`**: Utility to extract `$ref` IDs from various EIDO field formats.
    *   **`_attempt_geocode_and_update_store(self, text_to_geocode)`**:
        *   Calls the `AdvancedGeocodingService` to geocode textual location descriptions.
        *   If successful and confidence is sufficient (High/Medium), it updates the `local_geocoder`'s store (`data/geocoded_locations.json`) with the new coordinates.
    *   **`_extract_core_data_from_dict(self, eido_dict)`**:
        *   Crucial method for transforming a raw EIDO JSON dictionary into a standardized `ReportCoreData` Pydantic model.
        *   Extracts key fields: message ID, incident tracking ID, timestamp, incident type, descriptions (from notes/comments), location references, and agency references.
        *   Parses `locationByValue` (often XML containing PIDF-LO civic address or GML points) to extract coordinates, textual address, and ZIP code.
        *   **Stores a `location_narrative_for_geocoding` in `core_data.model_extra`** to be used if direct coordinates are missing.
    *   **`process_report_json(self, json_data)`**:
        *   Main entry point for processing validated EIDO JSON dictionaries or dictionaries generated from raw text.
        *   Calls `_extract_core_data_from_dict` to get `ReportCoreData`.
        *   **Enhanced Geocoding Step**: If `core_data.coordinates` are missing, it uses `location_narrative_for_geocoding` or the report description to call `_attempt_geocode_and_update_store`.
        *   Calls `find_match_for_report` to correlate the `ReportCoreData` with existing active incidents.
        *   If a match is found, updates the existing `Incident` object; otherwise, creates a new `Incident`.
        *   Uses `llm_interface` to generate/update the incident's summary and recommended actions.
        *   Saves the new or updated incident to the `IncidentStore`.
    *   **`process_alert_text(self, alert_text)`**:
        *   Main entry point for processing raw, unstructured alert text.
        *   Uses `llm_interface.split_raw_text_into_events` to divide text into individual event descriptions.
        *   For each event text, it calls `alert_parser.parse_alert_to_eido_dict` and then `process_report_json`.
    *   **`trigger_notification(self, incident, report_data, is_new)`**: Placeholder for future notification logic.
*   **Global Instance:** `eido_agent_instance = EidoAgent()` provides a singleton instance for use by the API.
*   **Purpose:** The central `EidoAgent` class orchestrates the data processing pipeline, from ingestion and geocoding to incident correlation and LLM-driven analysis. It is fully independent of the UI.

### 2.3. `agent/alert_parser.py`
*   **Purpose:** Responsible for converting raw alert text (representing a single event) into a structured, EIDO-like JSON dictionary using LLM assistance.
*   **Key Functions:**
    *   **`_generate_eido_compatible_dict(extracted_data)`**: Maps LLM-extracted data (type, timestamp, location, etc.) to a simplified EIDO dictionary structure. It constructs an XML string for `locationByValue`, embedding any coordinates or textual address.
    *   **`parse_alert_to_eido_dict(alert_text)`**: Calls `llm_interface.extract_eido_from_alert_text` and passes the result to `_generate_eido_compatible_dict`.
*   **Dependencies:** `agent.llm_interface.extract_eido_from_alert_text`.

### 2.4. `agent/llm_interface.py`
*   **Purpose:** Provides a unified interface for all interactions with Large Language Models (LLMs). It loads prompts from `agent/prompt_library.json` and supports multiple LLM providers.
*   **Key Functions & Logic:**
    *   **Loads prompts from `agent/prompt_library.json`** into a dictionary on startup.
    *   **`_get_llm_client(config)`**: Caches and returns LLM client instances.
    *   **`_call_llm(prompt, is_json_output)`**: The core function for making API calls to the configured LLM, handling JSON mode and provider-specific logic.
    *   **Task-Specific Functions**: `summarize_incident`, `recommend_actions`, `fill_eido_template`, `extract_eido_from_alert_text`, `split_raw_text_into_events`, `geocode_address_with_llm`, `extract_geolocatable_clues`. Each function formats a prompt from the loaded library and calls `_call_llm`.
*   **Dependencies:** `agent/prompt_library.json`, LLM client libraries, other project services.
*   **Purpose:** Provides a unified, decoupled interface for all interactions with LLMs. It is now fully independent of Streamlit and relies solely on the backend's configuration (`config/settings.py`), making it robust for deployment.
*   **Key Functions & Logic:**
    *   Loads prompts from `agent/prompt_library.json`.
    *   `_get_current_llm_config()`: Simply uses the backend's settings, ensuring no dependency on UI state.
    *   Handles API calls to different LLM providers ('google', 'openrouter', 'local').

### 2.5. `agent/prompt_library.json`
*   **Purpose:** A JSON file that stores all prompt templates used by the `llm_interface`. This separates prompt engineering from application logic, making the prompts easier to manage and modify without code changes.
*   **Key Aspects/Logic:**
    *   Contains a single JSON object.
    *   Keys are uppercase identifiers for prompts (e.g., `SUMMARIZE_INCIDENT`).
    *   Values are the multi-line prompt strings, with newlines represented as `\n`. Placeholders like `{history}` are kept as-is for runtime formatting.

### 2.6. `agent/matching.py`
*   **Purpose:** Implements the logic for correlating new incoming reports with existing active incidents to avoid duplicates.
*   **Key Functions:**
    *   **`calculate_similarity(core_data, incident)`**: Calculates a similarity score (0-1) based on time, external ID, location (coordinates), and content.
    *   **`find_match_for_report(core_data, incidents)`**: Iterates through active incidents, calls `calculate_similarity`, and returns the best match if its score is above `SIMILARITY_THRESHOLD`.
*   **Dependencies:** `data_models.schemas`, `config.settings`, `geopy`.

---

## 3. API Module (`api/`)

The `api/` module defines the FastAPI backend application, including its entry point and HTTP endpoints.

### 3.1. `api/__init__.py`
*   **Purpose:** An empty file that marks the `api` directory as a Python package.

### 3.2. `api/endpoints.py`
*   **Purpose:** Defines all the RESTful API endpoints for the EIDO Sentinel application.
*   **Key Endpoints:** Include `/api/v1/ingest`, `/api/v1/ingest_alert`, incident management routes, admin routes, EIDO generation, and local geocoding store management.
*   **Dependencies:** `fastapi`, `data_models.schemas`, `agent.agent_core.eido_agent_instance`, `services.storage.IncidentStore`, `config.settings`, `agent.llm_interface`, `services.local_geocoder`.
*   **Purpose:** Defines all RESTful API endpoints, serving as the interface between the frontend UI and the backend agent logic.

### 3.3. `api/main.py`
*   **Purpose:** The main entry point for the FastAPI backend application. Initializes the app, configures middleware, mounts static file serving, includes the API router, and manages application lifespan events.
*   **Dependencies:** `fastapi`, `uvicorn`, `config.settings`, `api.endpoints`, `services.database`.
*   **Purpose:** The main entry point for the FastAPI backend. Initializes the app, manages lifespan events (like DB init), and configures middleware.
*   **Key Aspects/Logic for Deployment:**
    *   **CORS Middleware**: Dynamically configured to allow requests from the deployed Streamlit frontend. It reads the `STREAMLIT_APP_URL` from its environment variables to securely add it to the list of allowed origins.

---

## 4. Configuration Module (`config/`)

This module is responsible for managing application settings.

### 4.1. `config/__init__.py`
*   **Purpose:** An empty file that marks the `config` directory as a Python package.

### 4.2. `config/settings.py`
*   **Purpose:** Defines and manages all application-level settings and configurations using Pydantic's `BaseSettings`.
*   **Key Class: `Settings(BaseSettings)`**: Defines fields for database URLs, API keys, LLM provider and model names, logging levels, incident matching thresholds, and **`geocoding_user_agent`**. Includes validators for critical settings.
*   **Global Instance:** `settings = Settings()`.
*   **Purpose:** Defines and validates all application settings using Pydantic's `BaseSettings`. It correctly loads configuration from environment variables, making it ideal for deployed environments.

---

## 5. Data Models Module (`data_models/`)

This module defines the Pydantic schemas used for data validation and structuring.

### 5.1. `data_models/__init__.py`
*   **Purpose:** An empty file that marks the `data_models` directory as a Python package.

### 5.2. `data_models/schemas.py`
*   **Purpose:** Defines the core Pydantic models (`ReportCoreData` and `Incident`).
*   **Key Classes:**
    *   **`ReportCoreData(BaseModel)`**: Includes `coordinates` (Tuple[float, float]) and `location_address`. `model_extra` is used to temporarily hold `location_narrative_for_geocoding`.
    *   **`Incident(BaseModel)`**: Consolidates multiple reports and includes aggregated `locations` (List of coordinates).
*   **Dependencies:** `pydantic`, `datetime`, `uuid`.
*   **Purpose:** Defines the core Pydantic models (`ReportCoreData`, `Incident`) that structure the application's data. `ReportCoreData` is configured with `extra='allow'` to support flexible fields like `location_narrative_for_geocoding`.

---

## 6. EIDO Templates (`eido_templates/`)

This directory stores JSON files that serve as templates for generating EIDO messages.

### 6.1. `eido_templates/traffic_collision.json`
### 6.2. `eido_templates/ucsd_burglary_template.json`
### 6.3. `eido_templates/ucsd_vegetation_fire_template.json`
*   **Purpose:** EIDO JSON templates with placeholders for use with the "EIDO Generator" feature.

---

## 7. Services Module (`services/`)

The `services/` module provides various backend services.

### 7.1. `services/__init__.py`
*   **Purpose:** Initializes the `services` package and exports key functions/classes.

### 7.2. `services/advanced_geocoding_service.py`
*   **Purpose:** Implements sophisticated, multi-step geocoding for textual narratives.
*   **Key Class: `AdvancedGeocodingService`**
    *   **`geocode_narrative(self, narrative_text)`**: Orchestrates clue extraction (LLM), geocoding of explicit addresses (Nominatim, LLM fallback), processing of named entities (local_geocoder, campus_geocoder, Nominatim POI), and fallback geocoding of the full narrative (LLM).
    *   Returns coordinates, confidence level (High, Medium, Low, None), method, and reasoning.
*   **Dependencies:** `agent.llm_interface`, `services.geocoding`, `services.campus_geocoder`, `services.local_geocoder`.

### 7.3. `services/campus_geocoder.py`
*   **Purpose:** Simple dictionary-based geocoder for known UCSD campus locations.
*   **Key Data:** `UCSD_NAMED_LOCATIONS` dictionary.

### 7.4. `services/database.py`
*   **Purpose:** Defines SQLAlchemy ORM models (`ReportCoreDataDB`, `IncidentDB`) and manages async database connections/sessions for PostgreSQL. `ReportCoreDataDB` stores `coordinates_lat` and `coordinates_lon`.
*   **Dependencies:** `sqlalchemy`, `config.settings`, `uuid`.

### 7.5. `services/eido_retriever.py`
*   **Purpose:** Implements RAG retrieval by loading an indexed EIDO OpenAPI schema and finding relevant chunks for LLM queries.
*   **Dependencies:** `services.embedding`, `config.settings`, `numpy`, `sklearn`. Index file `eido_schema_index.json` generated by `utils/rag_indexer.py`.

### 7.6. `services/embedding.py`
*   **Purpose:** Generates vector embeddings from text using SentenceTransformer models.
*   **Dependencies:** `sentence_transformers`, `config.settings`, `numpy`.

### 7.7. `services/geocoding.py`
*   **Purpose:** Standard geocoding using Nominatim (OpenStreetMap), with caching and rate limiting.
*   **Key Aspects:** Uses `GEOCODING_USER_AGENT` from settings.
*   **Dependencies:** `geopy`, `config.settings`.

### 7.8. `services/local_geocoder.py`
*   **Purpose:** Manages a local JSON file (`data/geocoded_locations.json`) for custom location-name-to-coordinate mappings. This file is updated by `AdvancedGeocodingService` (via `agent_core`) and can be managed via API/UI.
*   **Key Functions:** `_load_locations`, `_save_locations`, `get_coordinates_from_local_store`, `update_known_location`, `remove_known_location`.
*   **Dependencies:** `agent.llm_interface` (for LLM fallback).

### 7.9. `services/storage.py`
*   **Purpose:** Data Access Layer for persisting and retrieving incident/report data from the database, converting between Pydantic and SQLAlchemy models. `Incident` Pydantic models will now have their `locations` field populated with coordinates obtained through the enhanced geocoding pipeline.
*   **Dependencies:** `sqlalchemy`, `data_models.schemas`, `services.database`.

---

## 8. Static Assets (`static/`)

Contains static files for the project's landing/showcase page.

### 8.1. `static/css/styles.css`
### 8.2. `static/images/`
### 8.3. `static/videos/`
### 8.4. `static/index.html`
### 8.5. `static/js/3d-scene.js`
### 8.6. `static/js/main.js`
*   **Purpose:** Files for the static landing page, including HTML, CSS, JavaScript for 3D animations and general interactivity, and media assets.

---

## 9. User Interface Module (`ui/`)

Contains the Streamlit application for interactive demonstration.

### 9.1. `ui/__init__.py`
*   **Purpose:** An empty file that marks the `ui` directory as a Python package.

### 9.2. `ui/app.py`
*   **Purpose:** The main Streamlit application file. Creates the UI, handles user inputs, calls the FastAPI backend, and visualizes results.
*   **Map Tab**: Uses `pydeck` with `ScatterplotLayer` to pinpoint incident locations and `HeatmapLayer` to shade areas based on incident density. The data for the map comes from `Incident.locations`, which are populated by the agent after geocoding.
*   **Geocoding Tools Tab**: Allows manual management of the `data/geocoded_locations.json` file via API calls.
*   **Dependencies:** `streamlit`, `pandas`, `pydeck`, `requests`, `streamlit-ace`, `utils.ocr_processor`, `config.settings`, `data_models.schemas`.
*   **Purpose:** The main Streamlit application file. It is now **deployment-aware**.
*   **Key Aspects/Logic for Deployment:**
    *   **Environment Detection**: At startup, the app checks if it's running in a deployed environment (like Streamlit Cloud) by looking for `st.secrets['API_BASE_URL']`.
    *   **Dynamic API URL**: If deployed, it uses the `API_BASE_URL` from secrets. If running locally, it falls back to the URL from the local settings configuration. This allows the same codebase to work in both environments without changes.
    *   **API-Driven**: The UI interacts with the backend exclusively through API calls, ensuring a clean separation of concerns.

### 9.3. `ui/components.py`
*   **Purpose:** Placeholder/early-stage module for reusable Streamlit UI components. Currently not deeply integrated.

### 9.4. `ui/custom_styles.css`
*   **Purpose:** Custom CSS for the Streamlit application, providing a themed appearance.

---

## 10. Utilities Module (`utils/`)

Contains helper scripts and functions.

### 10.1. `utils/__init__.py`
*   **Purpose:** An empty file that marks the `utils` directory as a Python package.

### 10.2. `utils/helpers.py`
*   **Purpose:** Miscellaneous utility functions, currently for parsing XML snippets (e.g., PIDF-LO from EIDO `locationByValue`).

### 10.3. `utils/ocr_processor.py`
*   **Purpose:** Extracts text from images using Tesseract OCR.

### 10.4. `utils/rag_indexer.py`
*   **Purpose:** Script to build the RAG knowledge base index from the EIDO OpenAPI schema.

### 10.5. `utils/schema_parser.py`
*   **Purpose:** Helper to parse the EIDO OpenAPI schema (YAML) and format components for LLM consumption.