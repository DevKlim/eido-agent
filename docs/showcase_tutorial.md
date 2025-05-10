# EIDO Sentinel: Showcase & Comprehensive Tutorial

## 1. EIDO Sentinel: An Overview

EIDO Sentinel is a Proof-of-Concept (PoC) AI agent designed to revolutionize the way emergency communications and incident reports are processed. It leverages the power of Large Language Models (LLMs) and modern data processing techniques to provide a more efficient, intelligent, and responsive system for public safety.

**Key Capabilities:**

*   **Multi-Format Ingestion:** Seamlessly handles both standardized NENA EIDO JSON messages and unstructured raw alert text (e.g., CAD summaries, dispatch notes, SMS messages).
*   **LLM-Powered Parsing & Structuring:**
    *   Intelligently splits blocks of text containing multiple event descriptions into individual, processable reports.
    *   Extracts critical information (incident type, timestamp, location details including address and ZIP code, narrative description, source agency, external identifiers like CAD numbers) from unstructured text, transforming it into an EIDO-like JSON structure.
    *   Utilizes **Retrieval-Augmented Generation (RAG)** by querying an indexed version of the official EIDO OpenAPI schema. This provides the LLM with relevant context, significantly improving parsing accuracy and adherence to EIDO standards for generated data.
*   **Flexible EIDO JSON Processing:** Robustly extracts core data from various EIDO JSON structures, adeptly handling common variations and extracting crucial details like coordinates and ZIP codes from embedded XML (e.g., PIDF-LO) or free-text location fields.
*   **Intelligent Incident Correlation:** Matches incoming reports (whether from JSON or parsed text) to existing active incidents. Correlation is based on configurable parameters:
    *   Time proximity (within a defined window).
    *   Location proximity (using geodesic distance calculations between coordinates).
    *   Matching external identifiers (e.g., CAD incident numbers).
*   **AI-Driven Summarization & Action Recommendation:**
    *   Generates concise, evolving summaries of incidents as new information arrives, providing a clear and up-to-date overview.
    *   Suggests relevant next steps and recommended actions for dispatchers or responders based on the current incident summary and the latest report data.
*   **EIDO Generation Tool:** Facilitates the creation of structurally compliant EIDO JSON examples. Users can select a predefined template, provide a scenario description, and the LLM (augmented by RAG) will populate the template.
*   **Configurable LLM Backend:** Offers flexibility in choosing the LLM provider. Through the UI, users can dynamically switch between:
    *   Google Generative AI (e.g., Gemini models).
    *   OpenRouter (access to a wide array of models like GPT, Claude, Llama).
    *   Local LLMs (via an OpenAI-compatible API, e.g., Ollama, LM Studio).
*   **Interactive Dashboard (Streamlit):** A user-friendly web interface for:
    *   Data ingestion (uploading files, pasting JSON/text).
    *   Real-time LLM configuration for the session.
    *   Comprehensive incident visualization (filterable lists, interactive maps, dynamic charts).
    *   Detailed views of individual incidents, including summaries, recommended actions, and associated report history.
    *   Advanced filtering by incident type, status, and ZIP code.
    *   Automated warning generation based on filtered incidents.
    *   EIDO Explorer to view the original or LLM-generated JSON for any report.
    *   Access to the EIDO Generation tool.
    *   Administrative actions (e.g., clearing the incident store) and viewing processing logs.
*   **Basic API (FastAPI):** Provides RESTful endpoints for programmatic interaction, enabling integration with other systems. This includes endpoints for data ingestion and incident data retrieval.

EIDO Sentinel aims to reduce manual workload, improve situational awareness, and support faster, more informed decision-making in critical emergency response scenarios.

---

## 2. Tutorial: Using the EIDO Sentinel Demo Application (Streamlit UI)

This tutorial guides you through setting up and using the EIDO Sentinel interactive demo application.

### 2.1. Prerequisites

Before running the demo, ensure you have completed the following:

1.  **Installation:**
    *   Clone the EIDO Sentinel repository from GitHub.
    *   Run the `install_dependencies.sh` script (or `python -m pip install -r requirements.txt` after setting up a virtual environment manually). This installs all necessary Python packages.
        ```bash
        git clone https://github.com/LXString/eido-sentinel.git
        cd eido-sentinel
        ./install_dependencies.sh # Or manual venv setup + pip install