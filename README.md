# 🏥 MedQuery – AI‑Powered Medical Q&A System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com/)

> Ask any medical question – the system understands symptoms, explains diseases, lists causes and risk factors.

---

## ✨ Features

- **Symptom prediction** – describe what you feel, we suggest a possible disease.
- **Disease lookup** – overview, causes, risk factors (data from Mayo Clinic).
- **Intelligent routing** – an LLM (Ollama) decides which action to take.
- **Interactive chat UI** – built with Streamlit.
- **Full training pipeline** – scrapers, data augmentation, and model training included.

---

## 🧱 Architecture

![architecture](./docs/architecture.png)

*(Mermaid diagram above)*

The system uses:

- **Web scraper** – collects structured disease information from Mayo Clinic.
- **Data augmentation** – generates more training examples for symptom prediction.
- **Two prediction models**:
  - *Symptom → Disease*: TF‑IDF + RandomForest (lightweight, fast)
  - *Overview → Disease*: BERT (optional, for future expansion)
- **Fuzzy lookup** – exact/fuzzy matching of disease names using `fuzzywuzzy`.
- **Agent** – an Ollama LLM (e.g. `llama3`) classifies the user’s intent and calls the appropriate function.
- **Streamlit UI** – chat interface with memory.

---

## 🚀 Getting Started

### 1. Clone & install dependencies

```bash
git clone https://github.com/ahmedayad0168/Medical-Diagnosis-Project.git
cd Medical-Diagnosis-Project
pip install -r requirements.txt