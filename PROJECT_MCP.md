# EIDO Sentinel: Master Control Program & File Explanations

## Project Overview

**EIDO Sentinel** is an AI-powered platform designed to enhance emergency response by intelligently processing, correlating, and analyzing diverse emergency data streams. It leverages NENA EIDO standards, LLMs, and agentic AI principles to transform raw data into actionable, structured insights.

**Version:** 1.0.0 (as of this document)

## Table of Contents

1.  [Root Directory Files](#1-root-directory-files)
2.  [Agent Module (`agent/`)](#2-agent-module-agent)
3.  [API Module (`api/`)](#3-api-module-api)
4.  [Configuration Module (`config/`)](#4-configuration-module-config)
5.  [Data & Template Files](#5-data--template-files)
6.  [Data Models Module (`data_models/`)](#6-data-models-module-data_models)
7.  [Services Module (`services/`)](#7-services-module-services)
8.  [User Interface Module (`ui/`)](#8-user-interface-module-ui)
9.  [Utilities Module (`utils/`)](#9-utilities-module-utils)

---

## 1. Root Directory Files

- **`.env.example`**: A template for environment variables required for local development. Users must copy this to `.env` and fill in values like API keys.
- **`Dockerfile` & `docker-compose.yml`**: Define the containerized environment for the application, allowing the API backend and Streamlit UI to run as separate but connected services. The `Dockerfile` now includes a step to run the RAG indexer during the build process.
- **`entrypoint.sh`**: The script that runs inside the Docker container to start either the API (`api`) or the UI (`ui`) based on the command provided.
- **`run_all.sh` / `run_all.bat`**: Scripts for running the application stack locally without Docker.
- **`requirements.txt`**: Lists all Python dependencies for the project.

---

## 2. Agent Module (`agent/`)

The `agent/` module contains the core intelligence of EIDO Sentinel. It implements the primary data processing pipeline.

### **EIDO Generation Pipeline (from Raw Text)**

1.  **`agent_core.py` -> `process_alert_text`**: The entry point. It receives raw text and uses `llm_interface.split_raw_text_into_events` to break it into individual event chunks.
2.  **`alert_parser.py` -> `parse_alert_to_eido_dict`**: This is the heart of the new pipeline. For each event chunk:
    - It loads all templates from the `eido_templates/` directory.
    - It calls `llm_interface.choose_eido_template`, which asks an LLM to select the most suitable template file for the event text.
    - It then calls `llm_interface.fill_eido_template`. This powerful function uses another LLM call to populate the chosen template with data extracted from the event text. It leverages RAG context from the `eido_retriever` to understand schema fields better.
3.  **`agent_core.py` -> `_extract_core_data_from_dict`**: The generated EIDO dictionary from the parser is then fed into this method, which is the same one used for direct JSON ingestion. It normalizes the data into a `ReportCoreData` model.
4.  **`agent_core.py` -> `_process_core_data`**: The `ReportCoreData` is then processed for incident matching, summarization, and action recommendation, unifying the workflow.

### **Key Files in `agent/`**

- **`agent_core.py`**: The `EidoAgent` class orchestrates the entire workflow, from data ingestion (both JSON and raw text) to incident correlation (`matching.py`), LLM-driven analysis (`llm_interface.py`), and storage (`services/storage.py`).
- **`alert_parser.py`**: Implements the "choose-then-fill" pipeline for converting raw alert text into a structured EIDO dictionary.
- **`llm_interface.py`**: A unified interface for all interactions with LLMs. It handles API calls to different providers ('google', 'openrouter', 'local') and is fully independent of the UI.
- **`prompt_library.json`**: A library of all prompt templates used by `llm_interface.py`. This separation allows for easy prompt engineering without code changes. The `CHOOSE_EIDO_TEMPLATE` and `FILL_EIDO_TEMPLATE` prompts are central to the new pipeline.
- **`matching.py`**: Implements the logic for correlating new reports with existing active incidents based on time, location, content, and external IDs to prevent duplicates.

---

## 3. API Module (`api/`)

The `api/` module defines the FastAPI backend application.

- **`main.py`**: The entry point for the FastAPI server. It initializes the app, configures CORS middleware for UI communication, and includes the API router.
- **`endpoints.py`**: Defines all RESTful API endpoints, serving as the interface between the frontend UI and the backend agent logic. Key endpoints include `/ingest` (for JSON), `/ingest_alert` (for text), and various routes for incident management, geocoding, and the new **EIDO Template tools**.

---

## 4. Configuration Module (`config/`)

- **`settings.py`**: Defines and validates all application settings using Pydantic's `BaseSettings`. It correctly loads configuration from a `.env` file and environment variables, making it ideal for both local development and deployment.

---

## 5. Data & Template Files

- **`data/`**: A directory for persistent data.
  - `eido_sentinel_local.db`: The SQLite database file for local development.
  - `geocoded_locations.json`: A store for custom, manually-verified location-to-coordinate mappings, managed via the UI.
  - `ucsd_alerts.json`, `ucsd_alerts_simulated.txt`: Sample input data.
- **`eido_templates/`**: Contains various EIDO JSON files that serve as structured templates for the new generation pipeline. This directory is now managed by the **Template Editor** GUI.
  - **`EIDOContext.xml`**: A new, human-readable summary of the EIDO standard's concepts and components, used to provide high-level context to the RAG system.
- **`sample_eido/`**: Contains complete, sample EIDO JSON files for testing and demonstration.
- **`schema/openapi.yaml`**: The formal NENA EIDO OpenAPI specification. This file is crucial as it's used by `utils/rag_indexer.py` to build the knowledge base for the RAG system and by the **Template Editor** to display all available options.

---

## 6. Data Models Module (`data_models/`)

- **`schemas.py`**: Defines the core Pydantic models (`ReportCoreData`, `Incident`) that structure the application's internal data, ensuring type safety and validation.

---

## 7. Services Module (`services/`)

This module provides various backend services that support the agent's functions.

- **`storage.py`**: The Data Access Layer for saving and retrieving incident data from the database.
- **`database.py`**: Manages the database connection (SQLAlchemy) and defines the database table models.
- **`eido_retriever.py`**: Implements the RAG retrieval system. It uses an index file (`eido_schema_index.json`) to find relevant parts of the EIDO schema to provide as context to the LLM during template filling.
- **`embedding.py`**: A service for generating vector embeddings from text, used by the RAG system.
- **`geocoding.py`, `campus_geocoder.py`, `local_geocoder.py`**: A suite of geocoders. `geocoding.py` uses the public Nominatim service, `campus_geocoder` uses a hardcoded list of campus locations, and `local_geocoder` uses the user-managed `geocoded_locations.json`.
- **`advanced_geocoding_service.py`**: A sophisticated service that orchestrates the other geocoders and uses LLMs to geocode complex location descriptions from narratives.

---

## 8. User Interface Module (`ui/`)

Contains the Streamlit web application for interactive demonstration.

- **`app.py`**: The main Streamlit application file. It creates the UI, handles user inputs, calls the FastAPI backend, and visualizes the results. It now includes the **"Template Editor"** page for building new EIDO templates.
- **`custom_styles.css`**: Provides a custom, themed appearance for the Streamlit app.

---

## 9. Utilities Module (`utils/`)

Contains helper scripts and functions.

- **`rag_indexer.py`**: A crucial script that reads both `schema/openapi.yaml` (for technical details) and `eido_templates/EIDOContext.xml` (for conceptual overview), chunks the content, generates embeddings, and saves the resulting index to `services/eido_schema_index.json`. **This script must be run to enable the RAG functionality.**
- **`schema_parser.py`**: A helper used by the indexer and the new API endpoint to load and parse the OpenAPI schema.
- **`helpers.py`**: Contains miscellaneous utility functions, such as for parsing XML snippets found within EIDO fields.
- **`ocr_processor.py`**: A utility to extract text from images using Tesseract OCR, enabling ingestion from screenshots or scanned documents.
