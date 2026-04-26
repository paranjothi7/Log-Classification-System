# HyBERT-SOC

### A Hybrid Regex-BERT-LLM Framework for Automated Log Classification in Security Operations Centers

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![JIRA](https://img.shields.io/badge/JIRA-Integrated-0052CC)](https://www.atlassian.com/software/jira)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Classification Pipeline](#-classification-pipeline)
- [Model Specification](#-bert-model-specification)
- [Performance Results](#-performance-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dashboard](#-dashboard)
- [JIRA Integration](#-jira-integration)
- [Paper Results](#-paper-results)
- [Technology Stack](#-technology-stack)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**HyBERT-SOC** is an end-to-end intelligent log classification system designed for Security Operations Centers (SOCs). It addresses the critical challenge of manually triaging large volumes of heterogeneous log data by implementing a three-stage cascaded AI pipeline that combines the speed of rule-based matching, the accuracy of transformer-based deep learning, and the reasoning capability of Large Language Models.

Traditional SOC log monitoring relies on basic keyword matching and fixed log levels, missing critical operational patterns and failing to scale with modern infrastructure. HyBERT-SOC solves this by:

- Classifying raw logs into **9 fine-grained actionable categories**
- Processing **1000+ logs per minute** using concurrent batch execution
- Automatically creating **JIRA incident tickets** for critical alerts
- Providing a **real-time Streamlit dashboard** for monitoring and analytics

> 📄 **Paper:** *A Hybrid Regex-BERT-LLM Framework for Automated Log Classification in Security Operations Centers*

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **3-Stage Hybrid Pipeline** | Regex → BERT → LLM cascade with confidence-gated routing |
| **9-Class Classification** | Fine-grained SOC taxonomy beyond binary threat detection |
| **High Throughput** | 1000+ logs/min via ThreadPoolExecutor batch processing |
| **BERT Fine-Tuning** | bert-base-uncased fine-tuned on 2,410 real SOC logs |
| **LLM Fallback** | GROQ Llama3-70B for ambiguous logs via zero-shot prompting |
| **Real-time Dashboard** | 6-page Streamlit interface with Plotly visualizations |
| **JIRA Automation** | Auto-create tickets for Critical/High severity alerts |
| **DB Agnostic** | SQLite for development, PostgreSQL for production |
| **Result Export** | CSV, JSON, and database storage of all classifications |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                           │
│     CSV Batch      Real-time Stream      Streamlit UI    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           STAGE 1: REGEX CLASSIFIER                      │
│   9 handcrafted rules · confidence 0.95 · ~1 ms         │
└──────────────────────┬──────────────────────────────────┘
                       │ conf < 0.95
                       ▼
┌─────────────────────────────────────────────────────────┐
│           STAGE 2: BERT CLASSIFIER                       │
│   bert-base-uncased · threshold 0.75 · ~100 ms          │
└──────────────────────┬──────────────────────────────────┘
                       │ conf < 0.75
                       ▼
┌─────────────────────────────────────────────────────────┐
│           STAGE 3: LLM CLASSIFIER                        │
│   GROQ Llama3-70B · zero-shot JSON · ~2000 ms           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           BATCH PROCESSING MODULE                        │
│   ThreadPoolExecutor · 4 workers · 1000+ logs/min       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           DATABASE LAYER                                 │
│   SQLAlchemy ORM · SQLite (dev) · PostgreSQL (prod)     │
└──────────┬────────────────────────────┬─────────────────┘
           │                            │
           ▼                            ▼
┌──────────────────┐        ┌───────────────────────────┐
│  INTEGRATION     │        │  VISUALIZATION LAYER       │
│  JIRA REST API   │        │  Streamlit · 6 pages       │
│  Auto tickets    │        │  Plotly charts             │
└──────────┬───────┘        └────────────┬──────────────┘
           │                             │
           └──────────────┬──────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    OUTPUT                                │
│         CSV · JSON · Database · JIRA Ticket             │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| **File** | `synthetic_logs.csv` |
| **Total samples** | 2,410 |
| **Label classes** | 9 |
| **Train split** | 1,928 (80%) |
| **Test split** | 482 (20%) |
| **Sources** | ModernCRM, ModernHR, BillingSystem, AnalyticsEngine, ThirdPartyAPI, LegacyCRM |

### Class Distribution

| Label | Count | Percentage |
|---|---|---|
| HTTP Status | 1,017 | 42.2% |
| Security Alert | 371 | 15.4% |
| System Notification | 356 | 14.8% |
| Error | 177 | 7.3% |
| Resource Usage | 177 | 7.3% |
| Critical Error | 161 | 6.7% |
| User Action | 144 | 6.0% |
| Workflow Error | 4 | 0.2% |
| Deprecation Warning | 3 | 0.1% |

### Complexity Tier Distribution

| Tier | Count | Classifier |
|---|---|---|
| `regex` | 500 | Stage 1 — Regex |
| `bert` | 1,903 | Stage 2 — BERT |
| `llm` | 7 | Stage 3 — LLM |

---

## 🔀 Classification Pipeline

### Stage 1 — Regex Classifier (~1 ms)
Rule-based fast-path using 9 handcrafted regex patterns. Handles high-confidence, structurally distinct logs like HTTP access logs and nova compute metrics. Returns confidence of `0.95` on match.

### Stage 2 — BERT Classifier (~100 ms)
Fine-tuned `bert-base-uncased` transformer with a 9-class sequence classification head. Tokenizes logs to 128 tokens max. Only invoked when Regex confidence falls below `0.95`. Escalates to LLM if confidence is below `0.75`.

### Stage 3 — LLM Classifier (~2000 ms)
GROQ-hosted Llama3-70B invoked via zero-shot structured prompting. Returns JSON with category, severity, confidence, and reasoning. Used for highly ambiguous logs — less than 1% of total volume.

### Severity Mapping

| Category | Severity |
|---|---|
| Critical Error | Critical |
| Security Alert | High / Critical |
| Workflow Error | High |
| Error | Medium |
| Resource Usage | Medium |
| HTTP Status | Info |
| System Notification | Info |
| User Action | Info |
| Deprecation Warning | Low |

---

## 🤖 BERT Model Specification

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Model type | `BertForSequenceClassification` |
| Total parameters | ~110M |
| Number of labels | 9 |
| Max sequence length | 128 tokens |
| Training epochs | 4 |
| Train batch size | 16 |
| Eval batch size | 32 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Evaluation strategy | Per epoch |
| Best model selection | Minimum eval loss |
| Framework | HuggingFace Transformers 4.40+ |

---

## 📈 Performance Results

| Metric | Value |
|---|---|
| Overall accuracy | > 85% |
| Average confidence | > 90% |
| Regex latency | ~1 ms/log |
| BERT latency | ~100 ms/log |
| LLM latency | ~2000 ms/log |
| Batch throughput | 1000+ logs/min |
| Thread workers | 4 concurrent |
| Database write | ~50 ms/entry |

### Method Distribution

| Method | Logs Handled | Percentage |
|---|---|---|
| Regex | ~500 | 20.7% |
| BERT | ~1,896 | 78.7% |
| LLM | ~7 | 0.3% |
| Unknown | ~7 | 0.3% |

---

## 📁 Project Structure

```
HyBERT-SOC/
├── .env.example                          # Environment variable template
├── .gitignore                            # Git ignore rules
├── README.md                             # This file
├── run.bat                               # Windows one-click launch
│
├── resources/
│   └── synthetic_logs.csv                # 2,410 log dataset
│
├── src/
│   ├── app.py                            # Streamlit dashboard (6 pages)
│   ├── init_database.py                  # DB setup and seeder
│   ├── run_pipeline.py                   # Full pipeline execution
│   ├── test_model.py                     # Classifier accuracy test
│   ├── test_jira.py                      # JIRA connection test
│   ├── generate_paper_results.py         # Paper charts and tables
│   │
│   ├── processors/
│   │   ├── enhanced_processor.py         # Regex + BERT + LLM pipeline
│   │   └── high_performance_processor.py # Concurrent batch engine
│   │
│   ├── database/
│   │   ├── connection.py                 # SQLAlchemy engine
│   │   ├── models.py                     # ORM models and enums
│   │   └── service.py                    # CRUD and analytics queries
│   │
│   ├── integrations/
│   │   └── jira/
│   │       └── client.py                 # JIRA REST API client
│   │
│   ├── utils/
│   │   └── result_saver.py               # CSV, JSON, DB result saver
│   │
│   └── requirements.txt
│
├── training/
│   └── train_bert.py                     # BERT fine-tuning script
│
├── models/
│   └── bert_log_classifier/              # Saved after training
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── vocab.txt
│       ├── label_map.json
│       └── eval_report.json
│
├── results/                              # Generated by run_pipeline.py
│   ├── classified_synthetic_logs.csv
│   └── classified_synthetic_logs.json
│
├── paper_results/                        # Generated by generate_paper_results.py
│   ├── confusion_matrix.png
│   ├── category_distribution.png
│   ├── method_distribution.png
│   ├── confidence_by_category.png
│   ├── classification_report.json
│   └── performance_summary.json
│
└── tests/
    └── test_processors.py                # 32 pytest unit tests
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- PostgreSQL 12+ (optional — SQLite used by default)
- GROQ API key (free at [console.groq.com](https://console.groq.com))

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/HyBERT-SOC.git
cd HyBERT-SOC
```

### Step 2 — Install dependencies

```bash
cd src
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu --timeout 1000
pip install "numpy<2" --force-reinstall
pip install "transformers>=4.40.0" "accelerate>=0.26.0"
pip install -r requirements.txt
```

### Step 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Database
DATABASE_URL=sqlite:///./log_classification.db

# GROQ API (required for LLM fallback)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192

# BERT Model
BERT_MODEL_PATH=../models/bert_log_classifier
BERT_CONFIDENCE_THRESHOLD=0.75

# JIRA (optional)
JIRA_SERVER=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@gmail.com
JIRA_API_TOKEN=your_api_token
JIRA_PROJECT_KEY=SA
```

---

## 🚀 Usage

### Step 1 — Train BERT model

```bash
cd training
python train_bert.py
```

### Step 2 — Initialize database

```bash
cd ../src
python init_database.py
```

### Step 3 — Run classification pipeline

```bash
python run_pipeline.py
```

### Step 4 — Launch dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501**

### Step 5 — Run tests

```bash
cd ..
python -m pytest tests/test_processors.py -v
```

Expected: **32 passed**

### Step 6 — Generate paper results

```bash
cd src
python generate_paper_results.py
```

---

## 📊 Dashboard

The Streamlit dashboard provides 6 interactive pages:

| Page | Description |
|---|---|
| 🛡️ **Dashboard** | KPI cards — total logs, security alerts, critical errors, avg confidence + trend charts |
| 📋 **Log Classification** | CSV upload with batch processing, progress bar, and result download |
| 📊 **Analytics** | Category distribution and hourly volume trend charts |
| 🔍 **Log History** | Filterable log table by category and severity with CSV export |
| 🧪 **Single Log Test** | Test individual log messages with confidence gauge |
| ⚙️ **System Status** | Health checks for DB, GROQ API, and JIRA |

---

## 🔗 JIRA Integration

HyBERT-SOC automatically creates JIRA tickets for classified logs:

```python
# Triggered automatically for Critical and High severity logs
client.create_ticket(
    summary="[Security Alert] Multiple login failures detected",
    description="Raw log: Multiple bad login attempts on user 8538",
    category="Security Alert",
    severity="High",
)
# Creates: SA-1 with priority High, label: soc-auto
```

### Setup

1. Create free account at [atlassian.com](https://www.atlassian.com/software/jira/free)
2. Generate API token at [Atlassian security settings](https://id.atlassian.com/manage-profile/security/api-tokens)
3. Update `.env` with your `JIRA_SERVER`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY`

---

## 📄 Paper Results

Run `generate_paper_results.py` to produce all figures and tables:

| File | Used As | Description |
|---|---|---|
| `confusion_matrix.png` | Figure 3 | 9×9 classification confusion matrix |
| `category_distribution.png` | Figure 2 | Ground truth vs predicted distribution |
| `method_distribution.png` | Figure 4 | Regex vs BERT vs LLM usage pie chart |
| `confidence_by_category.png` | Figure 5 | Average confidence per class bar chart |
| `classification_report.json` | Table 2 | Per-class precision, recall, F1 |
| `performance_summary.json` | Table 1 | Overall accuracy, throughput, latency |

---

## 🛠️ Technology Stack

### AI / ML

| Component | Technology |
|---|---|
| Transformer model | HuggingFace `bert-base-uncased` |
| Deep learning | PyTorch 2.4+ |
| LLM inference | GROQ — Llama3-70B-8192 |
| Training framework | HuggingFace Trainer API |
| ML utilities | Scikit-learn 1.4+ |

### Backend

| Component | Technology |
|---|---|
| ORM | SQLAlchemy 2.0 |
| Database (dev) | SQLite |
| Database (prod) | PostgreSQL 12+ |
| Logging | Loguru |
| Environment | python-dotenv |

### Frontend

| Component | Technology |
|---|---|
| Dashboard | Streamlit 1.32 |
| Charts | Plotly 5.20 |
| Data processing | Pandas 2.2 |

### Integrations

| Component | Technology |
|---|---|
| Ticket automation | JIRA REST API v2 |
| LLM API | GROQ Cloud |

---

## 🧪 Test Coverage

```
tests/test_processors.py — 32 unit tests

TestHTTPStatus         (4 tests)  — nova WSGI, RCODE, status codes
TestSecurityAlert      (6 tests)  — brute force, login failures, bypass
TestCriticalError      (5 tests)  — system errors, disk faults, boot failure
TestResourceUsage      (4 tests)  — nova compute claims, memory limits
TestSystemNotification (4 tests)  — file uploads, backups, reboots
TestUserAction         (2 tests)  — login/logout, account creation
TestError              (2 tests)  — shard replication, task failures
TestEnhancedProcessor  (5 tests)  — full pipeline, batch, accuracy
```

---

## 🏆 Research Contributions

1. **Three-stage cascaded pipeline** combining deterministic, statistical, and generative AI in a unified framework
2. **Confidence-gated routing** minimising LLM API cost — LLM used for less than 1% of logs
3. **9-class fine-grained SOC taxonomy** enabling automated workflow routing beyond binary classification
4. **Database-agnostic analytics** with runtime detection of SQLite vs PostgreSQL
5. **End-to-end automation** from raw log ingestion to JIRA ticket creation
6. **High-throughput batch processing** achieving 1000+ logs/min on CPU hardware

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/my-feature`
3. Commit your changes — `git commit -m "Add my feature"`
4. Push to branch — `git push origin feature/my-feature`
5. Open a pull request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Paranjothi**
- Project: HyBERT-SOC Log Classification System
- Domain: AI / Cybersecurity / Security Operations Centers
- Paper: *A Hybrid Regex-BERT-LLM Framework for Automated Log Classification in Security Operations Centers*

---

<p align="center">
  Built with BERT · GROQ · Streamlit · SQLAlchemy · JIRA
</p>
