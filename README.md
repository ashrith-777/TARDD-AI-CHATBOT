# 🚀  TARDD Enterprise

> **Autonomous, Multimodal AI Customer Support Agent with Real-Time Telemetry, Long-Term Memory, Security Guardrails, and Proactive Escalation Workflows.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Groq Acceleration](https://img.shields.io/badge/Groq-GPT--OSS--120B%20%7C%20Whisper--v3-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

**TARDD Enterprise** bridges the gap between basic Q&A chatbots and enterprise-grade support operations. Built on top of **Groq-accelerated LLMs**, **Whisper speech-to-text**, and **Hindsight long-term memory**, TARDD provides instantaneous sub-second voice/text resolution while continuously measuring user frustration, predicting churn risk, defending against prompt injections, and executing warm hand-offs to human engineers when critical incidents arise.

---

## 🚀 Key Features

### 🎙️ 1. Dual-Mode Multimodal Interaction
* **Sub-Second Speech-to-Text (STT):** Powered by Groq's `whisper-large-v3-turbo` model for near-instantaneous browser voice transcription.
* **Dual Text & Audio Output:** Generates clear Markdown responses paired with low-latency MP3 speech synthesis using `gTTS` and streaming memory buffers (`io.BytesIO`).
* **Interactive Controls:** Toggle voice playback on/off dynamically or select from standard neutral accents.

### 📊 2. Observable AI & Telemetry Inspector Dashboard
* **Real-Time Frustration Scoring:** Scans incoming sentiment and keyword patterns to calculate dynamic frustration metrics (e.g., `12/100 Stable` vs. `88/100 Elevated`).
* **Automated Churn Risk Badges:** Real-time visual status badges flagging high-risk user interactions.
* **Hindsight Memory Inspector:** Side-panel display showcasing active recalled infrastructure memory and live memory write logs (`retain_status`).

### 🛡️ 3. Defense-in-Depth Security
* **Prompt Injection Guardrails:** Intercepts prompt injections and jailbreak vectors before reaching the primary model.
* **Zero-Fail API Sanitization:** Automated string sanitization routines to eliminate malformed key errors (`401 Unauthorized`).

### 🚨 4. Seamless Human-Agent Escalation (Warm Hand-Off)
* **Automatic Emergency Detection:** Automatically triggers when frustration spikes (`>80/100`) or when human support is explicitly requested.
* **AI Executive Briefing Payload:** Generates a structured hand-off briefing containing an executive problem summary, sentiment state, actions attempted, and active AWS/PostgreSQL infrastructure context.

### 📋 5. Post-Incident Productivity & CSAT Feedback
* **Proactive Ticket Summarizer:** Automatically generates structured post-mortem reports (Root Cause Analysis, Resolution Steps, Recommended Action Items).
* **1–5 Star Micro-Feedback Loop:** Embedded micro-CSAT rating buttons under every assistant response, logging ratings directly into long-term memory logs.

---

## 🏗️ System Architecture

```mermaid
graph TD
    %% User Interfaces
    subgraph UI["Frontend Layer (Streamlit)"]
        A[User: Voice / Text Input] --> B[Browser Mic / Chat Input]
    end

    %% Security & Ingestion
    subgraph Security["Security & Ingestion Engine"]
        B --> C[Security Guardrails]
        C -->|Passed| D[Groq Whisper-large-v3-turbo STT]
        C -->|Flagged| X[Security Intervention Alert]
    end

    %% Intelligence Core
    subgraph Core["Intelligence Core & Telemetry"]
        D --> E[Frustration & Sentiment Scanner]
        E --> F[Hindsight API / AWS Local Memory Fallback]
        F --> G[Groq GPT-OSS-120B / Qwen Engine]
    end

    %% Outputs & Enterprise Workflows
    subgraph Outputs["Dual Output & Operational Workflows"]
        G --> H[gTTS Audio Synthesis Buffer]
        G --> I[Inspector Panel: Real-Time Telemetry]
        G --> J[Warm Hand-Off Briefing & Ticket Summarizer]
        G --> K[CSAT 1-5 Star Micro-Feedback Loop]
    end

    %% Styling & Classes
    classDef security fill:#f87171,stroke:#b91c1c,color:#fff;
    classDef core fill:#38bdf8,stroke:#0284c7,color:#000;
    class X security;
    class G core;