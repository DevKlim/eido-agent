# EIDO Sentinel: Demo Plan & Presentation Guide 🚨

## Introduction

This document provides a step-by-step guide for demonstrating the capabilities of the **EIDO Sentinel** AI Incident Processor. It also includes key discussion points for presenting the project, its underlying AI concepts, benefits, and future potential.

**Goal:** Showcase how EIDO Sentinel leverages AI (specifically LLMs) to ingest, process, correlate, summarize, and visualize emergency incident reports from various sources (structured EIDO JSON and unstructured text).

**Audience:** Emergency response personnel, dispatch managers, IT staff, potential contributors, anyone interested in AI applications for public safety.

---

## I. EIDO Sentinel Overview

EIDO Sentinel is a **Proof-of-Concept (PoC)** AI agent designed to streamline the processing of emergency communications. Key features include:

1.  **Multi-Format Ingestion:** Handles standardized NENA EIDO JSON messages and unstructured alert text (e.g., CAD summaries, SMS).
2.  **LLM-Powered Parsing:** Uses Large Language Models (LLMs) to:
    *   Potentially split multi-event text into individual reports.
    *   Parse unstructured text into an EIDO-like JSON structure, extracting key fields.
3.  **Flexible EIDO JSON Processing:** Extracts core information from EIDO JSON, handling common variations.
4.  **RAG Augmentation:** Enhances LLM accuracy for parsing and template filling by retrieving relevant context from the official EIDO schema (requires indexing).
5.  **Incident Correlation:** Matches incoming reports to existing incidents based on time, location, and external IDs.
6.  **AI Summarization & Actions:** Generates evolving incident summaries and recommended actions using LLMs.
7.  **Configurable LLM Backend:** Supports Google Gemini, OpenRouter, and Local LLMs (Ollama/LM Studio via OpenAI API).
8.  **EIDO Generation Tool:** Creates structurally compliant EIDO examples using templates and LLM assistance.
9.  **Interactive Dashboard (Streamlit):** Provides visualization (List, Map, Charts), filtering, detailed views, warning generation, and EIDO exploration/generation tools.
10. **Basic API (FastAPI):** Offers endpoints for programmatic ingestion and retrieval.

---

## II. Running the Demo

Follow these steps to demonstrate the core functionalities of EIDO Sentinel.

### A. Prerequisites

1.  **Installation Complete:** Ensure all steps in the `README.md` under "Getting Started" -> "Installation" have been completed successfully.
2.  **Environment Variables (`.env`):**
    *   The `.env` file must exist and be configured with a valid `LLM_PROVIDER` and corresponding API keys/URLs/model names. **The demo requires a functional LLM backend for text parsing, summarization, actions, and generation.**
    *   `GEOCODING_USER_AGENT` must be set with a valid identifier.
3.  **(Recommended) RAG Index:** Ensure the RAG index has been built by running `python utils/rag_indexer.py`. This significantly improves LLM parsing/generation accuracy. Check for `services/eido_schema_index.json`.
4.  **(For Generator Demo) Template File:** Ensure at least one template file exists in the `eido_templates/` directory. For example, create `eido_templates/traffic_collision.json` with placeholders like `[TIMESTAMP_ISO_OFFSET]`, `[LOCATION_ADDRESS]`, `[DESCRIPTION]`, `[INCIDENT_UUID]`, etc.

### B. Starting the Application

1.  Open your terminal in the project's root directory (`eido-sentinel/`).
2.  Activate your virtual environment (if using one): `source venv/bin/activate` (or `venv\Scripts\activate` on Windows).
3.  Run the Streamlit application:
    ```bash
    ./run_streamlit.sh
    # Or: streamlit run ui/app.py
    ```
4.  Access the application in your browser (usually `http://localhost:8501`).

### C. Demo Flow (Step-by-Step)

*(Perform these steps within the Streamlit UI)*

1.  **Initial State & Setup:**
    *   **(Optional) Clear Previous Data:** Navigate to the **Admin Actions** expander in the sidebar and click **🗑️ Clear All Incidents**. Confirm the dashboard metrics reset to zero.
    *   **Configure LLM:** Open the **⚙️ Configure Agent** expander in the sidebar. Select your desired **LLM Provider** (e.g., `google`) and ensure the necessary API Key and Model Name are entered correctly for the *current session*. (Explain that this allows switching LLMs without restarting).

2.  **Ingestion - EIDO JSON:**
    *   Go to the **📥 Data Ingestion** section in the sidebar, **📄 EIDO JSON** tab.
    *   Under "Load Sample:", select a valid EIDO file (e.g., `Sample call transfer EIDO.json`).
    *   Click **🚀 Process Inputs**.
    *   *Expected Outcome:* A success message appears. The dashboard metrics update (Total Incidents: 1). The incident appears in the **🗓️ List** tab and as a point on the **🗺️ Map** tab.

3.  **Ingestion - Raw Text (Single Event):**
    *   Go to the **✍️ Raw Text** tab in the sidebar.
    *   Paste a simple, single-event alert text into the "Paste Alert Text" area. Example:
        ```
        UNIT ALERT: Engine 4 responding to report of smoke in building at 555 Oak Avenue, Springfield, ZIP 98765. Time: 14:32 PST. CAD ID: SFD24-00123. Possible kitchen fire.
        ```
    *   Click **🚀 Process Inputs**.
    *   *Expected Outcome:* Success message. Metrics update (Total Incidents: 2). A new incident appears in the List/Map.
    *   **Show Parsing:** Go to the **🔍 Details** tab, select the newly created incident (likely the most recent). Show the AI Summary. Then, navigate to the **📄 EIDO Explorer** tab, select the same incident, and select its associated report. Show the *generated* EIDO-like JSON in the Ace editor, demonstrating how the LLM parsed the text. Point out extracted fields like type, location, description, source (if parsed), and ZIP code.

4.  **Ingestion - Raw Text (Multi-Event):**
    *   Go back to the **✍️ Raw Text** tab. Clear the previous text.
    *   Paste text containing multiple distinct events. Example:
        ```
        15:10: MVA reported I-90 EB @ MP 15. Two vehicles, unknown injuries. Ref# ABC111. // 15:12: Update on ABC111: Confirmed non-injury. Tow truck requested. // 15:15: New call: Structure fire alarm, 123 Pine St, Redmond. Engine 1 dispatched. Ref# XYZ222
        ```
    *   Click **🚀 Process Inputs**.
    *   *Expected Outcome:* Processing message indicates multiple events might have been processed. Check the **📄 Processing Log** expander in the sidebar - look for messages about splitting the text and processing individual events. Metrics update. You might see one incident updated (ABC111) and one new incident created (XYZ222). Verify in the **🗓️ List** tab.

5.  **Ingestion - Raw Text (Update/Correlation):**
    *   Go back to the **✍️ Raw Text** tab. Clear the previous text.
    *   Paste text that clearly updates an *existing* incident (use details from one created earlier, e.g., the "555 Oak Avenue" fire). Example:
        ```
        UPDATE: Incident SFD24-00123 at 555 Oak Avenue. Fire knocked down. Requesting ventilation. Time: 14:55 PST.
        ```
    *   Click **🚀 Process Inputs**.
    *   *Expected Outcome:* Success message. Check the **📄 Processing Log** - look for messages indicating a match was found. The *Total Incidents* count should *not* increase. Go to the **🗓️ List** tab, find the "Structure Fire" incident, and note its "Reports" count increased and "Last Update" time changed. Go to the **🔍 Details** tab, select this incident, and show how the AI Summary has evolved to include the update. Check the "Associated Reports" expander to see multiple reports linked.

6.  **Dashboard Interaction & Filtering:**
    *   Go to the **🔎 Filter & Analyze Incidents** section below the metrics.
    *   **Filter by Type:** Select "Structure Fire" (or another type present). Show how the **🗓️ List**, **🗺️ Map**, and **📈 Charts** tabs update dynamically to show only matching incidents. Clear the filter.
    *   **Filter by Status:** Select "Active" or "Updated". Show the dynamic updates again. Clear the filter.
    *   **Filter by ZIP:** Select a ZIP code present in the data (e.g., "98765" from the sample text). Show the dynamic updates. Clear the filter.
    *   **Map Interaction:** Go to the **🗺️ Map** tab. Zoom and pan the map. Hover over points to show the tooltips with Incident ID, Type, and Status.
    *   **Details View:** Go to the **🔍 Details** tab. Select different incidents from the dropdown and show how the displayed Summary, Actions, and report history change.

7.  **Warning Generation:**
    *   Go to the **🔎 Filter & Analyze Incidents** section. Apply a filter (e.g., by ZIP code or Type) to narrow down the incidents.
    *   Navigate to the **📢 Warnings** tab.
    *   Choose a severity level (e.g., "Advisory").
    *   (Optional) Add a custom message.
    *   Click **📝 Generate Warning**.
    *   *Expected Outcome:* A formatted warning message appears in the text area below, summarizing the filtered incidents.

8.  **EIDO Generator Tool:**
    *   Navigate to the **📝 EIDO Generator** tab.
    *   **Select Template:** Choose a template file from the dropdown (e.g., `traffic_collision.json` - *remind audience this needs to be created beforehand*).
    *   **Enter Scenario:** Write a brief scenario description in the text area. Example:
        ```
        Two-car accident at the intersection of Maple St and 5th Ave around 11:45 AM EST. Minor injuries reported. Police unit P-12 is on scene. External CAD ID: CAD2024-55001. Location coordinates: 40.7128, -74.0060. Zip: 10007. Source: NYPD Dispatch.
        ```
    *   Click **✨ Generate EIDO from Template**.
    *   *Expected Outcome:* An EIDO JSON structure appears in the Ace editor below, with placeholders filled based on the scenario. Point out how the LLM populated fields like timestamp, location, description, IDs, etc., based on the text and template structure.
    *   Show the **📥 Download Generated EIDO** button.

9.  **(Optional) LLM Configuration Change:**
    *   Go to the **⚙️ Configure Agent** expander in the sidebar.
    *   Change the **LLM Provider** (e.g., from `google` to `local`, ensuring the local server is running and configured). Update API keys/URLs/models as needed.
    *   Go to the **✍️ Raw Text** tab, paste a simple alert text again.
    *   Click **🚀 Process Inputs**.
    *   *Expected Outcome:* Processing occurs using the *newly selected* LLM backend. Check the **📄 Processing Log** for confirmation of which provider/model was called. The resulting incident details (summary, actions) might differ slightly based on the model used.

---

## III. Presentation Points / Discussion Topics

Use these points to explain the project during or after the demo.

### A. Core Workflow

*   **Ingestion:** Agent receives data (EIDO JSON file/paste, Raw Text paste).
*   **Parsing/Structuring:**
    *   *JSON:* Directly extracts key fields using flexible parsing.
    *   *Text:* Uses LLM (optionally RAG-enhanced) to split multi-event text and parse single events into a structured, EIDO-like format (including coordinates, ZIP).
*   **Core Data Creation:** Standardizes extracted info into an internal `ReportCoreData` object.
*   **Geocoding:** Attempts to find coordinates via `geopy` if missing from the input but an address is present.
*   **Correlation:** Compares the new `ReportCoreData` against *active* incidents in the store using time proximity, location proximity (geodesic distance), and external ID matching.
*   **Incident Update/Creation:**
    *   *Match Found:* Adds the `ReportCoreData` to the existing `Incident`, updating timestamps, locations, ZIPs, report count, and status.
    *   *No Match:* Creates a *new* `Incident` object using the `ReportCoreData`.
*   **LLM Enhancement:**
    *   **Summarization:** Generates/updates a concise `summary` for the incident based on its full history and the latest report.
    *   **Actions:** Suggests `recommended_actions` based on the latest summary and report data.
*   **Storage:** Saves the created/updated `Incident` object to the in-memory store.
*   **Visualization:** Streamlit UI reads from the store to display lists, maps, charts, details, etc., applying user filters.
*   **Generation:** Uses LLM (optionally RAG-enhanced) to fill predefined EIDO templates based on user scenarios.

### B. The Role of AI/LLMs: Improvements & Benefits

*   **How AI Improves Current Structures:**
    *   **Automated Parsing:** Handles unstructured text (CAD notes, SMS, transcripts) that traditional systems struggle with, converting it into usable data.
    *   **Enhanced Correlation:** Goes beyond simple ID matching to correlate based on time/location proximity, catching related events that might be missed manually.
    *   **Intelligent Summarization:** Synthesizes information from multiple updates into a concise, evolving summary, reducing cognitive load.
    *   **Decision Support:** Provides actionable recommendations based on the current situation.
    *   **Flexibility:** Adapts to variations in EIDO JSON structure and unstructured text formats more readily than rigid rule-based systems.
    *   **Scalability:** Can potentially handle higher volumes of incoming data compared to purely manual processing.
*   **Key Benefits:**
    *   🚀 **Speed & Efficiency:** Reduces manual effort in reading, linking, and summarizing reports.
    *   💡 **Enhanced Situational Awareness:** Provides a unified, summarized view of incidents faster.
    *   🤔 **Improved Decision Support:** AI summaries and actions aid quicker, more informed decisions.
    *   📉 **Reduced Operator Load:** Automates tedious parsing and correlation tasks.
    *   🗺️ **Better Resource Allocation:** Insights from correlated data can inform deployment.
    *   ⚙️ **Standardization:** Promotes consistent information handling across input formats.

### C. LLM Options, RAG & Cost Considerations

*   **Cloud LLMs (Google Gemini, OpenRouter):**
    *   *Pros:* High quality models, fast inference, easy API access, managed infrastructure.
    *   *Cons:* Pay-per-token costs (can add up with volume), data privacy concerns (data sent to third-party API), potential vendor lock-in. (Mention different model tiers: Flash/Haiku vs Pro/Opus for cost/capability trade-off).
*   **Local LLMs (Ollama, LM Studio):**
    *   *Pros:* No direct API costs, enhanced data privacy/control, potential for fine-tuning.
    *   *Cons:* Requires significant local hardware (GPU RAM!), setup/maintenance effort, potentially slower inference, model quality varies.
*   **Open Source Models (Llama 3, Mistral, Phi-3, etc.):**
    *   *Pros:* Free to download/run (hardware permitting), rapidly improving quality, available via cheaper hosting (OpenRouter).
    *   *Cons:* Still requires infrastructure (local or hosted).
*   **RAG (Retrieval-Augmented Generation):**
    *   *Concept:* The agent retrieves relevant snippets from the EIDO OpenAPI schema *before* calling the LLM for parsing or template filling.
    *   *Benefit:* Provides the LLM with specific context about expected fields, types, and structures, leading to more accurate and compliant outputs without needing full fine-tuning. (Mention the `utils/rag_indexer.py` script).
*   **Cost-Saving / Alternatives:**
    *   Use smaller/faster models (e.g., Gemini Flash, Llama 3 8B) for less complex tasks like basic extraction.
    *   Implement caching for repeated LLM calls (if applicable).
    *   Hybrid approach: Use rules/regex for highly structured parts, LLM for ambiguity.
    *   Fine-tuning a smaller open-source model (advanced).

### D. Integrating External Tools & Data

*   The agent's capabilities can be vastly expanded by giving it access to external tools/APIs:
    *   **Geospatial Tools:**
        *   *GIS APIs (ESRI, Google Maps, OpenStreetMap):* Find nearby resources (hospitals, hydrants), check jurisdictions, analyze proximity to sensitive areas, get real-time traffic.
        *   *Reverse Geocoding:* Convert coordinates back to addresses.
    *   **Weather APIs (OpenWeatherMap, etc.):** Get current/forecast weather crucial for fires, HAZMAT, etc.
    *   **Communication Tools:**
        *   *Alerting Platforms (Everbridge, Twilio SMS):* Distribute generated warnings automatically.
        *   *Collaboration Platforms (Slack, Teams):* Push incident summaries or alerts.
    *   **Information Retrieval:**
        *   *Internal Databases:* Access agency SOPs, contact lists, resource status.
        *   *Web Search:* Find information on specific chemicals, locations, etc.
        *   *Social Media Monitoring (Caution!):* Potentially gather citizen reports (requires careful filtering/verification).

### E. Towards Agentic AI (Using Frameworks)

*   **Agentic AI:** Moving beyond simple input-output processing to systems that can plan, reason, and use tools autonomously to achieve goals.
*   **Frameworks (LangChain, LlamaIndex):** Provide building blocks for:
    *   **Tool Definition:** Making external APIs/functions usable by the LLM.
    *   **Planning:** Allowing the LLM to decide *which* tool(s) to use in sequence based on the incident context (e.g., "If HAZMAT, use Web Search for chemical info AND Weather API").
    *   **Execution & Memory:** Calling tools, processing results, and maintaining context over time.
*   **EIDO Sentinel Context:** Could enable the agent to automatically fetch traffic data for an MVA, check weather for a fire, or look up SOPs for a specific incident type.

### F. Future Development: "EIDO-Agentic AI"

*   **Proactive Monitoring:** Agent actively monitors sources (e.g., message queues, email) instead of just reacting to direct input.
*   **Multi-Modal Input:** Process voice transcripts (from radio/911), images (traffic cams, drones), or sensor data.
*   **Complex Reasoning/Planning:** Handle multi-step incident responses using LLM planning (e.g., ReAct patterns).
*   **Learning & Adaptation:** Incorporate user feedback to improve parsing, summarization, and recommendations over time (e.g., prompt refinement, potentially fine-tuning).
*   **Deeper Integration:** Tighter coupling with CAD, RMS, Alerting platforms for seamless data flow and action execution.
*   **Explainability (XAI):** Provide clearer justifications for *why* the agent made specific correlations, summaries, or recommendations.

### G. Relevant Technologies Mentioned

*   **Vector Databases (ChromaDB, Pinecone, Weaviate):** Essential for efficient RAG by storing and searching text embeddings (like the EIDO schema chunks).
*   **PydanticAI:** Library to directly integrate LLM capabilities (like parsing text into a structured model) with Pydantic data models. Could be an alternative for the text parsing logic.
*   **LLM Monitoring (LangSmith, Arize):** Tools to track LLM performance, costs, and debug issues in production.
*   **Deployment (Docker, Kubernetes, Cloud ML Platforms):** Standard tools for packaging and running AI applications reliably.

---

## IV. Key Takeaways

*   EIDO Sentinel demonstrates the potential of AI/LLMs to significantly **enhance emergency response efficiency and situational awareness**.
*   It can process **both structured EIDO JSON and unstructured text alerts**, unifying disparate information sources.
*   Features like **AI-powered parsing, correlation, summarization, and action recommendation** provide direct value to dispatchers and commanders.
*   The system is **configurable** (LLM backend) and **extensible** (potential for tool integration).
*   Concepts like **RAG** improve accuracy without costly fine-tuning.
*   Future development points towards more **proactive, multi-modal, and agentic capabilities**.

---

## V. Q&A / Further Information

*   Open floor for questions.
*   Refer to the project `README.md` for setup and technical details.
*   GitHub Repository: [Link to your repo]
*   License: MIT

---