# Requirement Classification Platform (v2)

> An AI-powered full-stack web application for classifying software requirements as **Functional Requirements (FR)** or **Non-Functional Requirements (NFR)** using multiple LLM providers, prompt engineering strategies, and an analytics dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Database Schema](#database-schema)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Application Pages](#application-pages)
- [API Endpoints](#api-endpoints)
- [Prompt Strategies](#prompt-strategies)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Overview

This platform allows software engineers and requirements analysts to classify requirement statements (user stories) automatically using five state-of-the-art LLM providers. Instead of relying on a single model, the platform enables **cross-model comparison** — surfacing agreement as confidence and disagreement as a signal of ambiguity in the requirement itself.

**Key capabilities:**
- Single and batch classification of requirements
- Side-by-side comparison of LLM outputs
- Semantic similarity-based deduplication and grouping
- API cost and token usage tracking
- Ambiguity detection in requirement language
- Annotation and human feedback loop
- Google OAuth + email/password authentication

---

## Features

| Feature | Description |
|---|---|
| Multi-LLM Classification | Groq, Gemini, Claude, Cohere, Mistral in one interface |
| Prompt Engineering | Zero-shot, Few-shot, Chain-of-Thought strategies |
| Batch Processing | Upload CSV datasets; real-time streaming progress |
| Similarity Grouping | Semantic cosine similarity groups near-duplicate NFRs |
| Ambiguity Detection | Flags vague or poorly structured requirements |
| Analytics Dashboard | Charts for FR/NFR split, NFR categories, latency trends |
| API Usage Dashboard | Token count, cost per request, success rate tracking |
| Annotation System | Human ground-truth labelling with model agreement tracking |
| Feedback Loop | Users can correct model classifications |
| Adaptive Cache | File-based cache with TTL and confidence-based revalidation |
| Google OAuth | Sign in with Google in addition to email/password |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-Login, Authlib |
| Database | PostgreSQL, SQLAlchemy ORM, psycopg2 |
| LLM APIs | Groq (LLaMA 3.1), Google Gemini 2.5 Pro, Anthropic Claude 3 Haiku, Cohere Command R+, Ollama (Mistral) |
| ML / NLP | scikit-learn, pandas, custom similarity & ambiguity modules |
| Frontend | Jinja2, Tailwind CSS (Glassmorphic theme), Vanilla JavaScript |
| Auth | Flask-Login, Werkzeug password hashing, Authlib (Google OAuth 2.0) |
| Dev Tools | Git, GitHub, VS Code, pgAdmin, python-dotenv |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│              User (Browser)                          │
│   Single / Batch / Analytics / API Dashboard        │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP
                   ▼
┌─────────────────────────────────────────────────────┐
│            Flask Application (app.py)               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Auth Routes │  │ Main Routes  │  │ API Routes│  │
│  │ login/signup│  │ /single      │  │ /api/*    │  │
│  │ Google OAuth│  │ /batch       │  │ analytics │  │
│  └─────────────┘  │ /analytics   │  │ usage     │  │
│                   │ /comparison  │  │ batch_runs│  │
│                   └──────────────┘  └───────────┘  │
│                           │                         │
│          ┌────────────────┼─────────────────┐       │
│          ▼                ▼                 ▼       │
│  ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │code_integration│ │similarity_ │ │ambiguity_    │ │
│  │.py           │ │grouping.py  │ │detector.py   │ │
│  │(LLM router)  │ │(dedup NLP)  │ │(NLP flags)   │ │
│  └──────┬───────┘ └─────────────┘ └──────────────┘ │
│         │                                           │
└─────────┼───────────────────────────────────────────┘
          │ API Calls
    ┌─────┴──────────────────────────────────────┐
    │              LLM Providers                  │
    │  Groq │ Gemini │ Claude │ Cohere │ Mistral  │
    └─────────────────────────────────────────────┘
          │ Results
          ▼
┌─────────────────────────────────────────────────────┐
│              PostgreSQL Database                     │
│  users │ batch_runs │ batch_results │ feedback       │
│  requirement_history │ annotations                  │
└─────────────────────────────────────────────────────┘
```

---

## Database Schema

| Table | Purpose |
|---|---|
| `users` | Stores user accounts; supports email/password and Google OAuth |
| `batch_runs` | Tracks each experiment run (model used, prompt strategy, total stories) |
| `batch_results` | Stores individual classification results with latency, confidence, category |
| `feedback` | Captures human corrections to model predictions |
| `requirement_history` | Tracks edits to requirement text over time |
| `annotations` | Human ground-truth labels for model evaluation |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ShashankRajput90/Requirement-Classification-UI.git
cd Requirement-Classification-UI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root (see [Environment Variables](#environment-variables) below).

### 4. Set Up PostgreSQL

```sql
CREATE DATABASE nfr_fr_db;
```

Then run the database initialisation script:

```bash
python init_db.py
```

### 5. Run the Application

```bash
python app.py
```

Open in browser: `http://localhost:5000`

---

## Environment Variables

Create a `.env` file with the following keys:

```env
# Flask
SECRET_KEY=your_secret_key

# Database
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nfr_fr_db

# LLM API Keys
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
COHERE_API_KEY=your_cohere_key
CLAUDE_API_KEY=your_anthropic_key

# Google OAuth (optional)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

> **Note:** The Mistral model runs locally via Ollama and does not require an API key. Install Ollama and pull the Mistral model separately.

---

## Application Pages

| Route | Page | Description |
|---|---|---|
| `/` | Home | Landing dashboard (login required) |
| `/single` | Single Classification | Classify one requirement at a time, view LLM outputs side by side |
| `/batch` | Batch Processing | Upload a CSV, select model and strategy, stream results in real time |
| `/analytics` | Analytics Dashboard | FR/NFR distribution, NFR category breakdown, latency charts |
| `/api_dashboard` | API Usage Dashboard | Token usage, cost per request, success rate per provider |
| `/comparison` | Model Comparison | Compare accuracy, precision, recall, F1 across models |
| `/calibration` | Calibration Analysis | Confidence vs. actual accuracy calibration view |
| `/login` | Login | Email/password or Google Sign-In |
| `/signup` | Sign Up | Create a new account |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/single` | Classify a single requirement (JSON body) |
| `POST` | `/batch` | Batch classify from uploaded CSV (streaming NDJSON response) |
| `GET` | `/api/analytics_data` | Classification metrics (total, FR, NFR, categories, latencies) |
| `GET` | `/api/batch_runs` | List of all batch experiment runs |
| `GET` | `/api/batch/status` | Live progress of current batch run |
| `GET` | `/api/technique_comparison` | Compare prompt strategies across batch runs |
| `GET` | `/api/compare_prompting` | Aggregated stats per prompting technique |
| `GET` | `/api/grouping` | Semantic similarity grouping of NFR results |
| `GET` | `/api/calibration` | Confidence calibration data for latest annotated run |
| `GET` | `/api/comparison-data` | Per-model accuracy metrics |
| `POST` | `/api/reset_batch` | Clear all batch results from the database |

---

## Prompt Strategies

The platform supports three prompt engineering strategies, selectable per run:

| Strategy | Description |
|---|---|
| **Zero-shot** | Direct classification prompt with no examples |
| **Few-shot** | Prompt includes 2–3 labelled examples before the target requirement |
| **Chain-of-Thought** | Prompt instructs the model to reason step-by-step before classifying |

Results from different strategies can be compared in the Analytics and Technique Comparison views.

---

## Project Structure

```
Requirement-Classification-UI/
├── app.py                          # Main Flask application (routes, logic, cache)
├── models.py                       # SQLAlchemy database models
├── code_integration.py             # LLM provider router (Groq, Gemini, Claude, Cohere, Mistral)
├── similarity_grouping.py          # Semantic similarity engine for deduplication
├── ambiguity_detector.py           # NLP-based ambiguity flagging
├── keyword_highlighter.py          # Highlights NFR-relevant keywords in UI
├── context_utils.py                # Domain context and RQI (Requirement Quality Index) utilities
├── evaluation.py                   # Model evaluation metrics
├── Batches.py                      # Batch experiment utilities
├── init_db.py                      # Database initialisation script
├── alter_db.py                     # Schema migration helper
├── check_db.py                     # DB connection diagnostics
├── requirements.txt                # Python dependencies
├── templates/                      # Jinja2 HTML templates
│   ├── index.html
│   ├── single.html
│   ├── batch.html
│   ├── analytics.html
│   └── ...
├── static/                         # Static assets (CSS, JS, images)
├── graphs/                         # Generated graph outputs
├── cache/                          # File-based classification cache (auto-generated)
├── balanced_user_stories.csv       # Sample dataset for testing
└── .env                            # Environment variables (not committed)
```

---

## Author

**Shashank Rajput**  
B.Tech Computer Science (Data Science), Meerut Institute of Technology  
GitHub: [ShashankRajput90](https://github.com/ShashankRajput90)

---

## License

Open source — add your preferred license.

---

## Acknowledgements

This project integrates the following open-source and commercial technologies:

- [Flask](https://flask.palletsprojects.com/) — Python web framework
- [PostgreSQL](https://www.postgresql.org/) — Relational database
- [Groq](https://groq.com/) — High-speed LLM inference
- [Google Gemini](https://ai.google.dev/) — Multimodal LLM
- [Anthropic Claude](https://www.anthropic.com/) — Structured AI outputs
- [Cohere](https://cohere.com/) — NLP and classification APIs
- [Ollama](https://ollama.com/) — Local LLM inference (Mistral)
- [scikit-learn](https://scikit-learn.org/) — ML evaluation metrics
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first CSS framework

---

⭐ If you found this project useful, consider starring the repository!
