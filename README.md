# EIDO Sentinel: Advancing Emergency Response with Agentic AI

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%26%20Streamlit-ff69b4)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.9.1-orange)](https://github.com/LXString/eido-sentinel)

**EIDO Sentinel is a proof-of-concept Agentic AI system designed to intelligently process emergency reports, manage NENA-compliant Emergency Incident Data Objects (EIDOs), and deliver actionable intelligence to enhance data-driven decisions in public safety.**

This project, developed by Kliment Ho (Research Intern, SDSC, UCSD) and Dr. Ilya Zaslavsky (Director, Spatial Information Systems Laboratory, SDSC, UCSD), leverages Large Language Models (LLMs) and aligns with NG9-1-1 standards to address the evolving emergency landscape.

**Access the Project:**

- **Main Showcase & Landing Page:** Served by FastAPI, typically at `http://localhost:8000` (or your deployed API base URL).
- **Interactive Demo Application:** Streamlit app, typically at `http://localhost:8501` (or your deployed Streamlit URL).

## Getting Started: Local Setup

Follow these steps to get EIDO Sentinel running on your local machine.

### 1. Prerequisites

- **Python 3.9+**
- **Git**
- **Tesseract OCR:** Required for the image ingestion feature.
  - **Windows:** Download from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page. During installation, make sure to check the option to add Tesseract to your system PATH.
  - **macOS:** `brew install tesseract`
  - **Linux (Ubuntu/Debian):** `sudo apt-get install tesseract-ocr`
- **(Optional) PostgreSQL Server:** For a more robust database. If not installed, the application will default to using a local SQLite file database, which requires no setup.
- **(Optional) Local LLM Server:** If you plan to use a local LLM, have a server like [Ollama](https://ollama.com/) running.

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/LXString/eido-sentinel.git
cd eido-sentinel
```

### 3. Install Dependencies

Create a virtual environment and install the required Python packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file to a new `.env` file. This file stores your secret keys and configuration.

```bash
cp .env.example .env
```

### 5. Run Database Migrations (for PostgreSQL)

If you are using PostgreSQL, initialize the database schema:

```bash
alembic upgrade head
```

For SQLite, the database file will be created automatically on first run.

### 6. Start the Applications

You can run both the FastAPI backend and Streamlit frontend simultaneously using the provided script, or start them individually.

#### Option A: Run All (Recommended)

```bash
./run_all.sh
```

This script will start the FastAPI backend and the Streamlit frontend in separate processes.

#### Option B: Run Individually

**Start FastAPI Backend:**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API and landing page at `http://localhost:8000`.

**Start Streamlit Frontend:**

```bash
streamlit run app/streamlit_app.py
```

Access the interactive demo at `http://localhost:8501`.

You are now ready to explore EIDO Sentinel!

## Project Vision: The Evolving Emergency Landscape

The emergency response domain faces increasing challenges:

- **Data Overload:** A deluge of information from diverse sources.
- **Need for Speed & Accuracy:** Critical decisions require rapid, precise intelligence.
- **NG911 Transition:** The shift towards IP-based infrastructure and richer data formats.

EIDO Sentinel aims to be a pivotal solution in this landscape by providing an **Agentic AI System** that:

- Monitors and processes diverse emergency reports.
- Manages NENA-compliant EIDOs (Emergency Incident Data Objects).
- Delivers actionable intelligence for enhanced situational awareness.

**EIDO Definition:** The Emergency Incident Data Object (EIDO) is a standard JSON format (NENA-STA-024.1.1-2025) developed by NENA to facilitate the sharing of emergency incident information across various systems and agencies. This project also considers the future concept of Incident Data Exchange (IDX).

## Core Features & Capabilities (v0.9.1)

EIDO Sentinel offers a robust set of features:

1.  **Rich Landing Page:** A professionally designed `index.html` (accessible via the FastAPI backend) showcasing the project's vision, capabilities, and call for collaboration.
2.  **Multi-Format Data Ingestion:**
    - NENA EIDO JSON (direct upload, paste, sample loading).
    - Unstructured raw alert text (e.g., CAD summaries, 911 transcripts, emails).
    - Text from images via OCR.
3.  **LLM-Powered Parsing & Structuring:**
    - Intelligently splits multi-event text into individual reports.
    - Extracts key information (type, time, location, ZIP, description, source, ID) from text.
    - Transforms text to EIDO-like JSON, utilizing **Retrieval-Augmented Generation (RAG)** with the EIDO OpenAPI schema for improved accuracy and compliance.
4.  **Robust EIDO JSON Processing:**
    - Extracts core data from complex EIDO JSON structures.
    - Handles embedded XML (e.g., PIDF-LO for civic addresses and GML for coordinates).
5.  **Intelligent Incident Correlation:**
    - Matches incoming reports to existing incidents based on time, location (contextual and standard geocoding), and external identifiers.
6.  **AI-Driven Summarization & Action Recommendation:**
    - Generates evolving, concise summaries of incidents.
    - Suggests actionable next steps for responders or dispatchers.
7.  **EIDO Generation Tool:**
    - Facilitates creating compliant EIDO JSON examples using predefined templates and LLM assistance based on scenario descriptions.
8.  **Configurable LLM Backend (Privacy First):**
    - Dynamically switch between:
      - Cloud LLMs (Google Gemini, OpenRouter for GPT/Claude models).
      - Local LLMs (Ollama, LM Studio compatible models like LLaMA) for on-premise, secure processing.
9.  **Interactive Demo Dashboard (Streamlit):**
    - User-friendly interface for data ingestion and LLM configuration (though LLM config is primarily backend-driven now).
    - Visualization of incidents: List, Map (PyDeck), Charts (Plotly).
    - Filtering and detailed views of incidents and associated reports.
    - Warning generation tool.
    - EIDO Explorer for viewing raw EIDO JSON.
10. **Contextual Geocoding:**
    - Goes beyond standard street addresses.
    - Understands landmarks and local names (e.g., "Geisel Library").
    - Integrates custom local datasets (e.g., UCSD campus locations).
    - Parses human descriptions of locations.
11. **API Backend (FastAPI):**
    - RESTful endpoints for programmatic ingestion (`/api/v1/ingest`, `/api/v1/ingest_alert`).
    - Endpoints for incident retrieval, status updates, and admin functions.
    - Serves the main landing page and static assets.
12. **Persistent Storage:**
    - Utilizes a PostgreSQL database for storing incidents and report core data, managed via SQLAlchemy.

## Workflow: Report to NENA-Compliant EIDO

1.  **Raw Report Ingested:** Input from 911 calls, dispatch summaries, or existing EIDO-JSON.
2.  **AI Parsing & Structuring:** Agent identifies key terms, location, type, and serializes to EIDO-JSON.
3.  **Dynamic Updates:** Constant feed of alerts updates existing incidents and warnings.
4.  **NENA EIDO Created/Managed:** API and tools enable analysis and generation of NENA-compatible alerts.

## Roadmap & Future Vision

We aim to significantly enhance data-driven decisions made by the current agent model. Key areas for future development include:

- **Advanced Geocoding Agent:**
  1.  **LLM + Map Search:** Geocode based on user input (location name) by refining it with an LLM, then performing a map search (e.g., Google Maps, Nominatim) to retrieve and store coordinates.
  2.  **Satellite Imagery + GIS + Multimodal LLM:** Pinpoint locations from user descriptions combined with satellite imagery analysis, using a multi-agentic structure for image retrieval, user comprehension, and GIS data integration.
- **Multi-Agentic System Conversion:**
  - **EIDO Agent:** Specialized for individual EIDO processing, validation, and enrichment.
  - **IDX Agent:** Focused on broader Incident Data Exchange, aggregating EIDOs, creating composite views, and simulating inter-agency communication.
- **EIDO Sentinel Refinements:**
  - **Historical Incident Reports Integration:** Analyze past data for trends, improved matching, and predictive insights.
  - **Real-time Data Feed Capabilities:** Enable continuous ingestion from live sources (IoT, CAD).
  - **Additional EIDO Wrappers:** Develop more utilities for converting various data formats to EIDO.
  - **Secure Access and Notifications:** Implement robust security and an intelligent notification system.
  - **Pilot Implementation:** Partner with agencies for real-world deployment and feedback.

## Call for Collaboration

We are actively seeking partners and collaborators to shape the future of public safety AI:

- **Workflow Integration:** How does EIDO Sentinel fit with your current workflows and NG911 plans?
- **Pain Point Identification:** What are your current challenges in report processing and alerting?
- **AI Automation Opportunities:** Which manual tasks are ripe for AI automation?
- **Data & Expertise Sharing:** Would you be interested in collaboration or providing anonymized/sample data fragments? Your insights are invaluable.

Please reach out to discuss potential partnerships or provide feedback.

## Troubleshooting

### Problem: The Streamlit App shows a blank page or an error on startup.

This is the most common issue after a fresh clone. It means the UI application crashed before it could render. Here is a checklist to solve it:

1.  **Is the Backend API Running?**

    - The Streamlit UI **must** connect to the FastAPI backend. You must start the backend first or use the `./run_all.sh` script which does it for you.
    - If you run `./run_api.sh` in one terminal, you should see logs from the Uvicorn server. Then run `./run_streamlit.sh` in a _second_ terminal.

2.  **Did you create and configure the `.env` file?**

    - The application needs this file for configuration (like API keys and settings).
    - Run `cp .env.example .env` and edit the new `.env` file.
    - **Crucially, you must change `GEOCODING_USER_AGENT`** to include your real email address.

3.  **Are all dependencies installed?**

    - Run `./install_dependencies.sh` or `pip install -r requirements.txt` again to be sure.

4.  **Did you build the RAG index?**

    - The agent relies on a search index of the EIDO schema.
    - Run `python utils/rag_indexer.py`. You should see a success message and a new file at `services/eido_schema_index.json`.

5.  **Check Terminal for Errors**
    - Look at the terminal where you started the backend (`run_api.sh` or `run_all.sh`). Are there any red error messages?
    - Look at the terminal where you started the frontend (`run_streamlit.sh`). The robust UI will now print specific error messages there to guide you.

By following this checklist, you should be able to resolve any startup issues.
