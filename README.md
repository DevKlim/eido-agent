# EIDO Sentinel API: AI-Powered Incident Intelligence

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%26%20Streamlit-ff69b4)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Proof-of-Concept AI agent for intelligent processing of emergency reports and enhanced data-driven decisions for public safety.**

EIDO Sentinel leverages AI, particularly Large Language Models (LLMs), to refine diverse data inputs into coherent, actionable intelligence. It aims to support faster, smarter decisions in critical situations, aligning with NG9-1-1 goals and NENA standards.

**Access the Project:**
*   **Main Showcase & Landing Page:** Served by FastAPI, typically at `http://localhost:8000`
*   **Interactive Demo Application:** Streamlit app, typically at `http://localhost:8501`

## Overview

In emergency response, timely and accurate information is paramount. EIDO Sentinel is an AI-powered system designed to:
*   **Ingest and Understand Data:** Process reports from various sources, including standardized NENA EIDO JSON, unstructured text alerts (CAD summaries, SMS), and even text from images via OCR.
*   **Structure and Correlate:** Transform raw data into structured, EIDO-compliant formats. Intelligently correlate new reports with existing incidents based on time, location, and other identifiers.
*   **Generate Insights:** Use LLMs to generate evolving incident summaries, recommend actions, and facilitate the creation of EIDO messages.
*   **Enhance Decision-Making:** Provide a clearer, consolidated view of events to support dispatchers, first responders, and command staff.

This project aims to create AI solutions for safety, addressing pain points in report processing, automating manual tasks, and preparing for a data-rich future in emergency communications.

## Core Features & Capabilities

EIDO Sentinel offers a robust set of features:

1.  **Rich Landing Page:** A professionally designed `index.html` (accessible at `http://localhost:8000`) showcasing the project's vision, capabilities, and call for collaboration.
2.  **Multi-Format Ingestion:** Handles NENA EIDO JSON and unstructured raw alert text.
3.  **LLM-Powered Parsing & Structuring:**
    *   Intelligently splits multi-event text into individual reports.
    *   Extracts key information (type, time, location, ZIP, description, source, ID) from text.
    *   Transforms text to EIDO-like JSON, using **Retrieval-Augmented Generation (RAG)** with the EIDO OpenAPI schema for improved accuracy.
4.  **Flexible EIDO JSON Processing:** Robustly extracts data from varied EIDO JSON, including embedded XML (PIDF-LO) for location details.
5.  **Intelligent Incident Correlation:** Matches incoming reports to existing incidents (time, location, external IDs).
6.  **AI-Driven Summarization & Action Recommendation:** Generates evolving summaries and suggests actionable next steps.
7.  **EIDO Generation Tool:** Facilitates creating compliant EIDO JSON examples using templates and LLM assistance.
8.  **Configurable LLM Backend:** Dynamically switch between Google Gemini, OpenRouter (GPT, Claude, etc.), and Local LLMs (Ollama, LM Studio) via the UI.
9.  **Interactive Demo Dashboard (Streamlit):** User-friendly interface for ingestion, LLM config, visualization (lists, maps, charts), filtering, warning generation, and EIDO exploration.
10. **API Backend (FastAPI):** RESTful endpoints for programmatic ingestion and data retrieval (under `/api/v1`).

## EIDO Sentinel Refinement: Vision & Roadmap

We aim to significantly enhance data-driven decisions made by the current agent model. Key areas of development include:

*   **Historical Incident Reports:**
    *   **Goal:** Integrate and analyze past incident data for trend identification, improved matching, and deeper contextual insights.
    *   **Impact:** Enable predictive analytics, resource optimization, and more informed strategic planning.
*   **Real-time Data Feed:**
    *   **Goal:** Enable continuous ingestion and processing of live data streams (e.g., IoT sensors, social media alerts, real-time CAD updates).
    *   **Impact:** Provide immediate situational awareness and faster response to developing events.
*   **Additional EIDO Wrappers:**
    *   **Goal:** Develop more sophisticated utilities for seamless conversion and integration of various data formats into NENA-compliant EIDO structures.
    *   **Impact:** Enhance interoperability with a wider range of data sources and systems.
*   **IDX Agent Capabilities:**
    *   **Goal:** Build specialized agent functionalities for robust Information Data Exchange (IDX) with diverse external public safety systems and platforms.
    *   **Impact:** Facilitate seamless data sharing and collaboration across agencies and jurisdictions.
*   **Secure Access and Notifications:**
    *   **Goal:** Implement robust security measures for data access (authentication, authorization) and develop an intelligent notification system for critical alerts and insights.
    *   **Impact:** Ensure data integrity, protect sensitive information, and deliver timely, relevant alerts to the right personnel.
*   **Pilot Implementation:**
    *   **Goal:** Deploy refined EIDO Sentinel capabilities in controlled pilot programs with partner agencies.
    *   **Impact:** Gather real-world feedback, validate effectiveness, and demonstrate operational value.

### Call for Collaboration

We are actively seeking partners and collaborators:
*   **How does this fit with your current workflows and NG911 plans?** We need input to ensure EIDO Sentinel aligns with real-world operational needs.
*   **What are your current pain points in report processing & alerting?** Your challenges can guide our AI development.
*   **Manual tasks ripe for AI automation?** Help us identify areas where AI can provide the most significant efficiency gains.
*   **Would you be interested in collaboration or can provide data fragments?** Anonymized or sample data is crucial for refining our models. Please reach out if you can assist.

## Project Structure

(Run `python print_struc.py` for an up-to-date tree)