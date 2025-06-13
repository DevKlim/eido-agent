# EIDO Sentinel: Advancing Emergency Response with Agentic AI

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%26%20Streamlit-ff69b4)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.9.1-orange)](https://github.com/LXString/eido-sentinel)

**EIDO Sentinel is a proof-of-concept Agentic AI system designed to intelligently process emergency reports, manage NENA-compliant Emergency Incident Data Objects (EIDOs), and deliver actionable intelligence to enhance data-driven decisions in public safety.**

This project, developed by Kliment Ho (Research Intern, SDSC, UCSD) and Dr. Ilya Zaslavsky (Director, Spatial Information Systems Laboratory, SDSC, UCSD), leverages Large Language Models (LLMs) and aligns with NG9-1-1 standards to address the evolving emergency landscape.

**Access the Project:**
*   **Main Showcase & Landing Page:** Served by FastAPI, typically at `http://localhost:8000` (or your deployed API base URL).
*   **Interactive Demo Application:** Streamlit app, typically at `http://localhost:8501` (or your deployed Streamlit URL).

## Project Vision: The Evolving Emergency Landscape

The emergency response domain faces increasing challenges:
*   **Data Overload:** A deluge of information from diverse sources.
*   **Need for Speed & Accuracy:** Critical decisions require rapid, precise intelligence.
*   **NG911 Transition:** The shift towards IP-based infrastructure and richer data formats.

EIDO Sentinel aims to be a pivotal solution in this landscape by providing an **Agentic AI System** that:
*   Monitors and processes diverse emergency reports.
*   Manages NENA-compliant EIDOs (Emergency Incident Data Objects).
*   Delivers actionable intelligence for enhanced situational awareness.

**EIDO Definition:** The Emergency Incident Data Object (EIDO) is a standard JSON format (NENA-STA-024.1.1-2025) developed by NENA to facilitate the sharing of emergency incident information across various systems and agencies. This project also considers the future concept of Incident Data Exchange (IDX).

## Core Features & Capabilities (v0.9.1)

EIDO Sentinel offers a robust set of features:

1.  **Rich Landing Page:** A professionally designed `index.html` (accessible via the FastAPI backend) showcasing the project's vision, capabilities, and call for collaboration.
2.  **Multi-Format Data Ingestion:**
    *   NENA EIDO JSON (direct upload, paste, sample loading).
    *   Unstructured raw alert text (e.g., CAD summaries, 911 transcripts, emails).
    *   Text from images via OCR.
3.  **LLM-Powered Parsing & Structuring:**
    *   Intelligently splits multi-event text into individual reports.
    *   Extracts key information (type, time, location, ZIP, description, source, ID) from text.
    *   Transforms text to EIDO-like JSON, utilizing **Retrieval-Augmented Generation (RAG)** with the EIDO OpenAPI schema for improved accuracy and compliance.
4.  **Robust EIDO JSON Processing:**
    *   Extracts core data from complex EIDO JSON structures.
    *   Handles embedded XML (e.g., PIDF-LO for civic addresses and GML for coordinates).
5.  **Intelligent Incident Correlation:**
    *   Matches incoming reports to existing incidents based on time, location (contextual and standard geocoding), and external identifiers.
6.  **AI-Driven Summarization & Action Recommendation:**
    *   Generates evolving, concise summaries of incidents.
    *   Suggests actionable next steps for responders or dispatchers.
7.  **EIDO Generation Tool:**
    *   Facilitates creating compliant EIDO JSON examples using predefined templates and LLM assistance based on scenario descriptions.
8.  **Configurable LLM Backend (Privacy First):**
    *   Dynamically switch between:
        *   Cloud LLMs (Google Gemini, OpenRouter for GPT/Claude models).
        *   Local LLMs (Ollama, LM Studio compatible models like LLaMA) for on-premise, secure processing.
9.  **Interactive Demo Dashboard (Streamlit):**
    *   User-friendly interface for data ingestion and LLM configuration (though LLM config is primarily backend-driven now).
    *   Visualization of incidents: List, Map (PyDeck), Charts (Plotly).
    *   Filtering and detailed views of incidents and associated reports.
    *   Warning generation tool.
    *   EIDO Explorer for viewing raw EIDO JSON.
10. **Contextual Geocoding:**
    *   Goes beyond standard street addresses.
    *   Understands landmarks and local names (e.g., "Geisel Library").
    *   Integrates custom local datasets (e.g., UCSD campus locations).
    *   Parses human descriptions of locations.
11. **API Backend (FastAPI):**
    *   RESTful endpoints for programmatic ingestion (`/api/v1/ingest`, `/api/v1/ingest_alert`).
    *   Endpoints for incident retrieval, status updates, and admin functions.
    *   Serves the main landing page and static assets.
12. **Persistent Storage:**
    *   Utilizes a PostgreSQL database for storing incidents and report core data, managed via SQLAlchemy.

## Workflow: Report to NENA-Compliant EIDO

1.  **Raw Report Ingested:** Input from 911 calls, dispatch summaries, or existing EIDO-JSON.
2.  **AI Parsing & Structuring:** Agent identifies key terms, location, type, and serializes to EIDO-JSON.
3.  **Dynamic Updates:** Constant feed of alerts updates existing incidents and warnings.
4.  **NENA EIDO Created/Managed:** API and tools enable analysis and generation of NENA-compatible alerts.

## Roadmap & Future Vision

We aim to significantly enhance data-driven decisions made by the current agent model. Key areas for future development include:

*   **Advanced Geocoding Agent:**
    1.  **LLM + Map Search:** Geocode based on user input (location name) by refining it with an LLM, then performing a map search (e.g., Google Maps, Nominatim) to retrieve and store coordinates.
    2.  **Satellite Imagery + GIS + Multimodal LLM:** Pinpoint locations from user descriptions combined with satellite imagery analysis, using a multi-agentic structure for image retrieval, user comprehension, and GIS data integration.
*   **Multi-Agentic System Conversion:**
    *   **EIDO Agent:** Specialized for individual EIDO processing, validation, and enrichment.
    *   **IDX Agent:** Focused on broader Incident Data Exchange, aggregating EIDOs, creating composite views, and simulating inter-agency communication.
*   **EIDO Sentinel Refinements:**
    *   **Historical Incident Reports Integration:** Analyze past data for trends, improved matching, and predictive insights.
    *   **Real-time Data Feed Capabilities:** Enable continuous ingestion from live sources (IoT, CAD).
    *   **Additional EIDO Wrappers:** Develop more utilities for converting various data formats to EIDO.
    *   **Secure Access and Notifications:** Implement robust security and an intelligent notification system.
    *   **Pilot Implementation:** Partner with agencies for real-world deployment and feedback.

## Call for Collaboration

We are actively seeking partners and collaborators to shape the future of public safety AI:
*   **Workflow Integration:** How does EIDO Sentinel fit with your current workflows and NG911 plans?
*   **Pain Point Identification:** What are your current challenges in report processing and alerting?
*   **AI Automation Opportunities:** Which manual tasks are ripe for AI automation?
*   **Data & Expertise Sharing:** Would you be interested in collaboration or providing anonymized/sample data fragments? Your insights are invaluable.

Please reach out to discuss potential partnerships or provide feedback.

## Getting Started

(Instructions for local setup, dependencies, and running the application will be detailed here. For now, refer to `requirements.txt` and `run_all.sh` or individual component startup commands.)

1.  **Prerequisites:**
    *   Python 3.9+
    *   PostgreSQL server
    *   Tesseract OCR (for image ingestion)
    *   Access to an LLM (Cloud API key or local LLM server like Ollama)
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/LXString/eido-sentinel.git
    cd eido-sentinel