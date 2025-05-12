# EIDO Sentinel: Demo Plan & Presentation Guide

## Introduction

This document provides a step-by-step guide for demonstrating the capabilities of the **EIDO Sentinel** AI Incident Processor. It also includes key discussion points for presenting the project, its underlying AI concepts, benefits, and future potential.

**Goal:** Showcase how EIDO Sentinel leverages AI (specifically LLMs) to ingest, process, correlate, summarize, and visualize emergency incident reports from various sources (structured EIDO JSON, unstructured text, and text from images via OCR).

**Audience:** Emergency response personnel, dispatch managers, IT staff, potential contributors, anyone interested in AI applications for public safety.

---

## I. EIDO Sentinel Overview

EIDO Sentinel is a **Proof-of-Concept (PoC)** AI agent designed to streamline the processing of emergency communications. Its core strength lies in transforming diverse data inputs into coherent, actionable intelligence.

It adeptly handles standardized NENA EIDO JSON messages, unstructured alert text (such as CAD summaries or SMS), and can even extract text from images using **Optical Character Recognition (OCR)**. Using Large Language Models (LLMs), EIDO Sentinel parses this information, even splitting multi-event texts into individual reports and structuring raw text into an EIDO-like format. This process is enhanced by Retrieval-Augmented Generation (RAG), which consults the official EIDO schema for improved accuracy. The system can also leverage a **campus-specific geocoder** to identify coordinates for named locations (e.g., "Geisel Library UCSD") mentioned in descriptions, complementing traditional address geocoding.

The system intelligently correlates new reports with existing incidents based on time, location (including campus-specific named places), and external IDs. It then uses AI to generate evolving incident summaries and recommend actions. Users can configure various LLM backends (Google Gemini, OpenRouter, Local LLMs) via an interactive Streamlit dashboard. This dashboard also provides comprehensive visualization tools, filtering capabilities, and an EIDO generation utility. A FastAPI backend supports these functions and offers an API for programmatic integration.

---

## II. Running the Demo

Follow these steps to demonstrate the core functionalities of EIDO Sentinel.

### A. Prerequisites

1.  **Installation Complete:** Ensure all steps in the `README.md` under "Getting Started" -> "Installation" have been completed successfully. This includes installing Tesseract OCR for the OCR feature.
2.  **Environment Variables (`.env`):**
    *   The `.env` file must exist and be configured with a valid `LLM_PROVIDER` and corresponding API keys/URLs/model names. **The demo requires a functional LLM backend for text parsing, summarization, actions, and generation.**
    *   `GEOCODING_USER_AGENT` must be set with a valid identifier (e.g., `EidoSentinelDemo/1.0 (your.email@example.com)`).
3.  **(Recommended) RAG Index:** Ensure the RAG index has been built by running `python utils/rag_indexer.py`. This significantly improves LLM parsing/generation accuracy. Check for `services/eido_schema_index.json`.
4.  **(For Generator Demo) Template File:** Ensure at least one template file exists in the `eido_templates/` directory. For example, `eido_templates/traffic_collision.json`.

### B. Starting the Application

1.  Open your terminal in the project's root directory (`eido-sentinel/`).
2.  Activate your virtual environment (if using one): `source venv/bin/activate` (or `venv\Scripts\activate` on Windows).
3.  Run the combined launch script (recommended):
    ```bash
    ./run_all.sh