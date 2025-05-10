# EIDO Sentinel: AI Incident Processor

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%26%20Streamlit-ff69b4)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🚨 **Proof-of-Concept AI agent for processing emergency reports.**

EIDO Sentinel demonstrates how AI, particularly Large Language Models (LLMs), can enhance the processing of emergency communications. It features a rich, animated landing page and an interactive Streamlit demo application.

**Access the Project:**
*   **Main Showcase & Landing Page:** Served by FastAPI, typically at `http://localhost:8000`
*   **Interactive Demo Application:** Streamlit app, typically at `http://localhost:8501`

## Key Features

EIDO Sentinel offers a robust set of features:

1.  **Rich Landing Page:** A professionally designed `index.html` with CSS styling and JavaScript animations, serving as the main entry point and showcase for the project.
2.  **Multi-Format Ingestion:** Handles standardized NENA EIDO JSON messages and unstructured raw alert text (e.g., CAD summaries, dispatch notes, SMS messages) via the Streamlit demo and API.
3.  **LLM-Powered Parsing & Structuring:**
    *   Intelligently splits blocks of text containing multiple event descriptions into individual, processable reports.
    *   Extracts critical information (incident type, timestamp, location details including address and ZIP code, narrative description, source agency, external identifiers like CAD numbers) from unstructured text, transforming it into an EIDO-like JSON structure.
    *   Utilizes **Retrieval-Augmented Generation (RAG)** by querying an indexed version of the official EIDO OpenAPI schema. This provides the LLM with relevant context, significantly improving parsing accuracy and adherence to EIDO standards.
4.  **Flexible EIDO JSON Processing:** Robustly extracts core data from various EIDO JSON structures, adeptly handling common variations and extracting crucial details like coordinates and ZIP codes from embedded XML (e.g., PIDF-LO) or free-text location fields.
5.  **Intelligent Incident Correlation:** Matches incoming reports (whether from JSON or parsed text) to existing active incidents based on configurable parameters: time proximity, location proximity (geodesic distance), and external IDs.
6.  **AI-Driven Summarization & Action Recommendation:**
    *   Generates concise, evolving summaries of incidents as new information arrives, providing a clear and up-to-date overview.
    *   Suggests relevant next steps and recommended actions for dispatchers or responders based on the current incident summary and the latest report data.
7.  **EIDO Generation Tool:** Facilitates the creation of structurally compliant EIDO JSON examples. Users can select a predefined template, provide a scenario description, and the LLM (augmented by RAG) will populate the template.
8.  **Configurable LLM Backend:** Offers flexibility in choosing the LLM provider. Through the Streamlit UI, users can dynamically switch between:
    *   Google Generative AI (e.g., Gemini models).
    *   OpenRouter (access to a wide array of models like GPT, Claude, Llama).
    *   Local LLMs (via an OpenAI-compatible API, e.g., Ollama, LM Studio).
9.  **Interactive Demo Dashboard (Streamlit):** A user-friendly web interface for data ingestion, LLM configuration, incident visualization (filterable lists, interactive maps, dynamic charts), detailed views, advanced filtering (by type, status, ZIP code), automated warning generation, EIDO exploration, and administrative actions.
10. **API Backend (FastAPI):** Provides RESTful endpoints for programmatic interaction (under `/api/v1`) and serves the main landing page.

## Project Structure