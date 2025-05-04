**1. Updated `README.md`**

```markdown
# EIDO Sentinel: AI Incident Processor

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%26%20FastAPI-ff69b4)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) <!-- Add a LICENSE file -->

🚨 **Proof-of-Concept AI agent for processing emergency reports.**

This project demonstrates how AI, including Large Language Models (LLMs), can process emergency communications. It focuses on:
1.  Ingesting and processing standardized NENA EIDO messages (JSON format), extracting core information flexibly.
2.  Ingesting **unstructured alert text** (e.g., CAD summaries, SMS, transcripts) and using an LLM to:
    *   Potentially split text containing multiple events into individual reports.
    *   Parse each report into an EIDO-like JSON dictionary.
3.  Correlating incoming reports (from JSON or text) to existing incidents or creating new ones.
4.  Generating incident summaries and recommended actions using LLMs.
5.  Providing visualization and interaction via a Streamlit dashboard and basic API endpoints.
6.  Generating structurally compliant EIDO JSON examples using templates and LLM assistance.

**Current Capabilities:**

*   **Multi-Format Ingestion:** Handles EIDO JSON (single/list) and raw alert text.
*   **LLM-Powered Parsing:** Uses LLMs to split multi-event text blocks and extract key fields (type, time, location, description, source, ZIP code, external ID) from unstructured text into an EIDO-like structure.
*   **Flexible EIDO JSON Processing:** Parses key fields from provided EIDO JSON dictionaries, handling common variations.
*   **RAG-Augmented LLM Calls:** Enhances LLM parsing and template filling with relevant context retrieved from the official EIDO OpenAPI schema for improved accuracy and compliance awareness. (Requires indexing step).
*   **Incident Correlation:** Basic matching based on time, location proximity, and external IDs.
*   **AI Summarization & Actions:** Generates evolving incident summaries and recommended actions using a configured LLM.
*   **Configurable LLM Backend:** Supports Google Generative AI (Gemini), OpenRouter, and local LLMs (via OpenAI-compatible servers like LM Studio/Ollama) through UI configuration.
*   **EIDO Generation Tool:** Fills predefined EIDO templates with scenario details using LLM assistance to create structurally compliant examples.
*   **Interactive Dashboard (Streamlit):**
    *   Ingest data via file upload, text paste, or samples.
    *   Configure LLM settings for the session.
    *   View incident lists, details, summaries, and actions.
    *   Filter incidents by type, status, ZIP code.
    *   Visualize locations on an interactive map (PyDeck Heatmap/Scatterplot).
    *   Explore basic incident trends and status charts.
    *   Generate simple warning text based on filtered incidents.
    *   Explore original EIDO JSON data associated with reports using an embedded code editor.
    *   Generate new EIDO examples using the Generator tool.
*   **Basic API (FastAPI):** Endpoints for ingestion (`/ingest`, `/ingest_alert`) and incident retrieval.
*   **In-Memory Storage:** Stores incident data during runtime.

## Project Structure

```
./
├── EIDO-JSON/              # Cloned NENA EIDO JSON Repo (contains Schema/)
├── agent/                  # Core AI agent logic (core, parser, llm, matching)
├── api/                    # FastAPI backend
├── config/                 # Configuration (settings.py)
├── data_models/            # Pydantic schemas (internal: ReportCoreData, Incident)
├── eido_templates/         # Manually created EIDO JSON templates for generator
├── sample_eido/            # Example EIDO JSON input files
├── services/               # Supporting services (storage, geocoding, embedding, retriever)
├── tests/                  # Unit/integration tests (to be expanded)
├── ui/                     # Streamlit frontend application (app.py)
├── utils/                  # Helper functions (schema_parser, rag_indexer)
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
├── install_dependencies.sh # Install script (Linux/macOS)
├── run_api.sh              # Run FastAPI script
├── run_streamlit.sh        # Run Streamlit script
└── README.md               # This file
```

## Getting Started

### Prerequisites

*   Python 3.9+ & Pip
*   Git
*   (Optional) Virtual environment tool (`venv`)
*   API Key for your chosen LLM provider (Google, OpenRouter) OR a running local LLM server (LM Studio, Ollama). Required for most AI features.

### Installation

1.  **Clone this repository:**
    ```bash
    git clone <your-repo-url>
    cd eido-sentinel
    ```
2.  **Clone the NENA EIDO-JSON repository:**
    ```bash
    git clone https://github.com/NENA911/EIDO-JSON.git
    ```
3.  **Create/Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Set up Environment Variables:**
    *   Copy `.env.example` to `.env`.
    *   Edit `.env` and provide:
        *   `LLM_PROVIDER`: `google`, `openrouter`, `local`, or `none`.
        *   API keys/models/URLs based on the chosen provider (e.g., `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `LOCAL_LLM_API_BASE_URL`, `LOCAL_LLM_MODEL_NAME`).
        *   `GEOCODING_USER_AGENT`: **Required.** A descriptive user agent with contact info (e.g., `"MyEidoApp/1.0 (myemail@domain.com)"`).
        *   Other optional settings (matching thresholds, log level).

6.  **(Optional but Recommended for RAG) Build Schema Index:**
    *   Run the indexer script once:
        ```bash
        python utils/rag_indexer.py
        ```
    *   This creates `services/eido_schema_index.json`, which the retriever uses to augment LLM prompts.

### Running the Application

*   **Run the Streamlit UI:**
    ```bash
    ./run_streamlit.sh
    # Or manually: streamlit run ui/app.py
    ```
    Access the dashboard (usually `http://localhost:8501`). Configure LLM settings in the sidebar expander. **If using LM Studio, ensure the "OpenAI API" preset is selected and the server is running.**

*   **Run the FastAPI API (Optional):**
    ```bash
    ./run_api.sh
    # Or manually: uvicorn api.main:app --reload --host <host> --port <port>
    ```
    Access API docs (usually `http://localhost:8000/docs`).

## Usage (Streamlit UI)

1.  **Configure Agent:** Use the sidebar expander (`⚙️ Configure Agent`) to select your LLM provider (Google, OpenRouter, Local, None) and enter necessary API keys, model names, or URLs for the current session.
2.  **Ingest Data:** Use the sidebar tabs (`📄 EIDO JSON` or `✍️ Raw Text`) to upload files, paste JSON/text, or load samples.
3.  **Process:** Click `🚀 Process Inputs`. The agent will process the data using the configured LLM. Check logs and status messages for details.
4.  **Explore Dashboard:** Use the main tabs (`🗓️ List`, `🗺️ Map`, `📈 Charts`, `🔍 Details`, `📢 Warnings`, `📄 EIDO Explorer`, `📝 EIDO Generator`) to view, filter, analyze incidents, generate warnings, explore raw EIDO, or generate new EIDO examples.
5.  **Generate EIDO:** Use the `📝 EIDO Generator` tab: select a template from `eido_templates/`, describe a scenario, and click `✨ Generate EIDO from Template`.

## Contributing / License

Contributions welcome via Pull Requests. Licensed under the MIT License.
```

---

**2. Discussion: Next Steps, Tools, AI Role, etc.**

Here’s a breakdown addressing your points:

**A. Demo Plan to Exhaust Agent Capabilities**

A comprehensive demo should showcase the end-to-end workflow and flexibility:

1.  **Setup:**
    *   Start with an empty incident store (`Clear All Incidents`).
    *   Configure different LLM backends sequentially (e.g., start with Google, then switch to Local LM Studio).
2.  **Ingestion Variety:**
    *   **EIDO JSON:** Process a valid sample EIDO file (e.g., `Sample call transfer EIDO.json`). Show it appears in the list/map.
    *   **Raw Text (Single Event):** Process a simple, single-event alert text (e.g., "Car fire NB I-5 near exit 10, Engine 2 responding"). Show the generated incident, summary, and actions. Use the EIDO Explorer to view the *generated* EIDO-like JSON.
    *   **Raw Text (Multi-Event):** Process a text block containing 2-3 distinct events (like the example used for testing splitting). Show that *multiple* incidents might be created/updated. Check logs to verify splitting.
    *   **Raw Text (Update):** Process text that clearly updates an existing incident (e.g., using the same CAD ID or mentioning the same location shortly after). Show that it correctly matches and updates the incident (report count increases, summary changes).
3.  **Dashboard Interaction:**
    *   **Filtering:** Apply filters (Type, Status, ZIP) and show how the List, Map, and Charts views update dynamically.
    *   **Map:** Zoom/pan the PyDeck map, hover over points for tooltips.
    *   **Details:** Select different incidents in the Details view, examine summaries, actions, associated reports, and original EIDO JSON.
    *   **Warnings:** Filter to a specific ZIP code or incident type, generate a warning text.
4.  **EIDO Generator:**
    *   Select a template (e.g., `traffic_collision.json` - *you'll need to create this*).
    *   Write a simple scenario description.
    *   Generate the EIDO JSON and show the output in the Ace editor. Download the file.
5.  **Configuration Change:** Switch the LLM provider in the settings (e.g., from Google to Local) and re-process a raw text alert to show the system uses the new backend.

**B. Tools for Aid & Agentic Protocols**

The agent could be significantly enhanced by integrating external tools and data sources, moving towards more autonomous ("agentic") behavior:

*   **Geospatial Tools:**
    *   **Reverse Geocoding:** Convert coordinates back to addresses or points of interest (using `geopy` or other services).
    *   **GIS APIs:** Integrate with ESRI ArcGIS APIs, Google Maps APIs, or open GIS data (like OpenStreetMap Overpass API) to:
        *   Find nearby resources (hospitals, fire stations, hydrants).
        *   Determine jurisdiction boundaries.
        *   Analyze proximity to sensitive locations (schools, critical infrastructure).
        *   Get real-time traffic information.
    *   **Weather APIs:** (OpenWeatherMap, WeatherAPI.com) Fetch current/forecasted weather for incident locations (critical for fires, floods, HAZMAT).
*   **Communication Tools:**
    *   **Alerting Platforms:** Integrate with platforms like Everbridge, Alertus, or even basic SMS gateways (Twilio) to *distribute* the generated warnings.
    *   **Messaging/Collaboration:** Push summaries or alerts to platforms like Slack, Microsoft Teams, or dedicated incident command software.
*   **Information Retrieval:**
    *   **Internal Knowledge Bases:** Connect to agency databases (SOPs, contact lists, resource availability).
    *   **Web Search:** Allow the agent to search the web for supplemental information (e.g., details about a specific chemical involved in a HAZMAT incident).
    *   **Social Media Monitoring:** (Use with caution due to noise/privacy) APIs like Twitter's could potentially provide citizen reports or situational context near an incident location.
*   **Agentic Implementation:** Use frameworks like **LangChain** or **LlamaIndex**. These provide structures for:
    *   **Tool Definition:** Defining the external APIs/functions the LLM can use.
    *   **Planning:** Enabling the LLM to decide *which* tool to use based on the incident context (e.g., "If incident type is 'Wildfire', use Weather API and GIS API for nearby resources").
    *   **Execution:** Calling the chosen tool and incorporating the results back into the incident summary or recommended actions.

**C. How LLMs/AI Improve Current Structures & Cheaper Alternatives**

*   **Improvements:**
    *   **Speed & Efficiency:** Drastically reduces time spent manually reading, correlating, and summarizing alerts, especially from unstructured sources.
    *   **Correlation:** Automatically links related updates that might be missed manually, providing a unified view.
    *   **Consistency:** Applies standard parsing and summarization logic, reducing human variability.
    *   **Accessibility:** Parses diverse input formats (text, potentially voice transcripts in the future) into a structured internal representation.
    *   **Insight Generation:** Summaries and action recommendations provide immediate decision support.
    *   **Scalability:** Can handle high volumes of incoming reports more effectively than manual processing alone.
*   **Cheaper Alternatives / Cost Considerations:**
    *   **API Costs:** Services like Google Gemini and OpenRouter charge per token (input + output). Costs scale with usage volume and model complexity. Flash/Haiku models are cheaper than Pro/Opus models.
    *   **Local LLMs:**
        *   **Pros:** No direct API costs, enhanced privacy/data control, potential for fine-tuning on specific data.
        *   **Cons:** Requires significant hardware (especially GPU RAM for larger models like 30B+), setup/maintenance effort, potentially slower inference than optimized cloud APIs, model quality might vary.
    *   **Open Source Models:** Models like Llama 3, Mistral, Qwen, Phi-3 are free to download and run locally (subject to hardware) or via cheaper hosting providers on OpenRouter/etc. Their quality is rapidly improving and often rivals proprietary models for specific tasks like extraction.
    *   **Fine-tuning:** Fine-tuning a smaller open-source model specifically for EIDO extraction or summarization could yield high performance at lower inference cost (but requires data and expertise).
    *   **Rule-Based Systems:** For *very* structured input, traditional rule-based parsing (regex, keyword matching) is cheapest but brittle and fails on unstructured/varied text.
    *   **Hybrid Approach:** Use rules/keywords for initial triage or simple extraction, falling back to LLMs for complex cases.

**D. Why This Helps & Next Stage of Development**

*   **Why It Helps (Benefits):**
    *   **Enhanced Situational Awareness:** Provides dispatchers/commanders with faster, correlated, and summarized incident views.
    *   **Improved Decision Support:** AI-generated summaries and action recommendations aid in quicker, more informed decisions.
    *   **Reduced Operator Load:** Automates tedious tasks like parsing text alerts and correlating updates.
    *   **Better Resource Allocation:** Insights from correlated data and potential tool use (GIS, traffic) can inform resource deployment.
    *   **Standardization:** Promotes consistent handling of information, regardless of input format.
*   **Next Stage (EIDO-Agentic AI):**
    *   **Proactive Monitoring & Action:** Agent doesn't just react to input but monitors sources (if configured) and potentially initiates actions (e.g., "High-priority fire near school detected, automatically retrieve school contact info and draft evacuation advisory").
    *   **Multi-Modal Input:** Process voice (transcripts), images (from drones/traffic cams), or sensor data alongside text/JSON.
    *   **Complex Reasoning & Planning:** Use LLM planning capabilities (e.g., ReAct, Plan-and-Execute) to handle multi-step incident responses involving multiple tools.
    *   **Learning & Adaptation:** Implement feedback loops where user corrections (e.g., fixing a parsed location, validating a summary) improve the agent's future performance (potentially via prompt refinement or model fine-tuning).
    *   **Deeper Integration:** Tighter coupling with CAD systems, alerting platforms, and incident command software for seamless data flow and action execution.
    *   **Explainability:** Provide clearer explanations for *why* the agent made a certain summary, recommendation, or correlation decision.

**E. Local LLMs, Codeless, PydanticAI, Popular Tools**

*   **Local LLMs:** (As discussed above) Great for privacy and cost control if you have the hardware. Tools like **Ollama** and **LM Studio** make setup much easier by providing an OpenAI-compatible API server. Key trade-off is performance vs. hardware cost/capability.
*   **Codeless/Low-Code Setups:** Platforms like **Zapier**, **Make (Integromat)**, or **n8n** allow connecting different APIs and services with minimal coding, often using visual interfaces.
    *   *Relevance:* Could be used for *simpler* workflows around EIDO Sentinel (e.g., "When a new incident is created via API, send a Slack notification"), but building the core agent logic (parsing, correlation, LLM interaction, state management) typically requires custom code as implemented in this project. They are generally not suited for building the agent itself but can help with *integrating* its inputs/outputs.
*   **PydanticAI:** A library specifically designed to add LLM-powered features *directly to Pydantic models*. You could define a Pydantic model and ask PydanticAI+LLM to:
    *   Instantiate the model from unstructured text.
    *   Validate data using LLM reasoning.
    *   Generate data conforming to the model.
    *   *Relevance:* It could be an *alternative* way to implement the text-to-EIDO-like-structure parsing (`extract_eido_from_alert_text` + `alert_parser`). Instead of a generic LLM call followed by manual dict creation, you might define a Pydantic model for the extracted fields and use PydanticAI to populate it directly from the text. It's worth exploring for potentially cleaner parsing logic but adds a specific dependency.
*   **Popular AI Tools in Modern Setups:**
    *   **Agent Frameworks:** **LangChain** & **LlamaIndex** are dominant. They provide abstractions for prompts, LLM connections, document loading, indexing (including RAG), tool use, memory, and agent execution loops. Integrating EIDO Sentinel's logic into one of these frameworks could streamline development, especially for adding tools and complex reasoning.
    *   **Vector Databases:** **ChromaDB** (easy local start), **Pinecone**, **Weaviate**, **Qdrant**, **Milvus**. Essential for efficient RAG implementation by storing and searching text chunk embeddings.
    *   **LLM Monitoring/Observability:** **LangSmith** (from LangChain), **Arize AI**, **Weights & Biases**. Crucial for tracking LLM calls, evaluating performance, debugging prompts, and monitoring costs in production.
    *   **Deployment/Infra:** **Docker** (containerization), **Kubernetes** (orchestration), Cloud platforms (**AWS SageMaker**, **GCP Vertex AI**, **Azure ML**) for hosting models and applications, **Modal Labs** (serverless GPU).

This detailed breakdown should provide a solid foundation for your demo, future development planning, and understanding the broader context of AI in this domain.