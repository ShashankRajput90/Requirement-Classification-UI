# Requirement Classification Platform (v2)

An advanced **AI-powered platform for classifying software requirements** as **Functional Requirements (FR)** or **Non-Functional Requirements (NFR)** using multiple LLM providers, prompt engineering techniques, and analytics dashboards.

This version introduces **database-backed analytics, API usage tracking, cost monitoring, and batch experimentation** to evaluate models and prompting strategies.

---

## Key Features

### Multi-Provider LLM Integration

The platform supports multiple AI providers to classify requirements:

* **Groq**

  * LLaMA 3.1
  * DeepSeek / GPT-OSS models
* **Google Gemini**

  * Gemini 2.5 Pro
* **Cohere**

  * Command R+
* **Anthropic**

  * Claude 3 Haiku
* **Local Models**

  * Mistral via Ollama

This enables **cross-model benchmarking and experimentation**.

---

## Prompt Engineering Strategies

The system allows experimenting with different prompting methods:

* **Zero Shot**
* **Few Shot**
* **Chain of Thought**

These strategies help evaluate how prompting affects classification performance.

---

## Analytics Dashboard

The analytics dashboard provides real-time insights including:

### Classification Metrics

* FR vs NFR distribution
* NFR category breakdown
* Latency performance

### Prompting Technique Analysis

Compare results between:
* Zero-shot
* Few-shot
* Other prompting strategies

### Batch Experiment Tracking

Each batch run records:
* Model used
* Prompting technique
* Total stories processed
* Average latency
* Classification distribution

---

## API Usage & Cost Dashboard

A dedicated **API Usage Dashboard** tracks LLM consumption.

### Usage Metrics

* Total API calls
* Total tokens used
* Average cost per request
* Success rate

### Cost Calculation

Costs are calculated using provider pricing models:

```
Cost = (Prompt Tokens × Input Rate) +
       (Completion Tokens × Output Rate)
```

All usage data is stored in the database for historical analysis.

---

## Batch Processing

Users can upload CSV datasets for large-scale classification.

Features:

* CSV upload support
* Automatic column detection
* Batch run tracking
* Real-time progress monitoring
* Historical batch results

---

## Database Integration

The platform uses **PostgreSQL** for persistent storage.

### Core Tables

**users**

* Authentication and account tracking

**batch_runs**

* Tracks experiment runs
* Stores model and prompting strategy

**batch_results**

* Stores classification results
* Token usage
* Latency
* Cost per request

This enables **full experiment reproducibility and analytics**.

---

## Authentication System

The application includes a login system with:

* User signup
* Password hashing
* Session management
* Protected routes

Each user's experiments and analytics are isolated.

---

## System Architecture

```
Frontend (HTML / JS Dashboard)
        │
        ▼
Flask Backend (API Layer)
        │
        ├── LLM Integration Layer
        │       ├── Groq
        │       ├── Gemini
        │       ├── Cohere
        │       ├── Claude
        │       └── Local Mistral
        │
        ├── Prompt Engineering Engine
        │
        └── PostgreSQL Database
                ├── Users
                ├── Batch Runs
                └── Batch Results
```

---

## 🛠 Installation

### 1 Clone Repository

```
git clone https://github.com/ShashankRajput90/Requirement-Classification-UI.git
cd Requirement-Classification-UI
```

---

### 2 Install Dependencies

```
pip install -r requirements.txt
```

---

### 3 Configure Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
COHERE_API_KEY=your_key
CLAUDE_API_KEY=your_key
```

---

### 4 Setup PostgreSQL

Create a database:

```
CREATE DATABASE nfr_fr_db;
```

Update database credentials in `app.py`.

---

### 5 Run Application

```
python app.py
```

Open:

```
http://localhost:5000
```

---

## Application Pages

| Page                  | Description                                    |
| --------------------- | ---------------------------------------------- |
| Single Classification | Classify individual user stories               |
| Batch Processing      | Upload datasets for large-scale classification |
| Analytics Dashboard   | View experiment insights                       |
| API Usage Dashboard   | Monitor token usage and cost                   |
| Model Comparison      | Compare LLM performance                        |
| History               | View past classifications                      |

---

## API Endpoints

### Classification

```
POST /single
```

Classify a single requirement.

---

### Batch Processing

```
POST /batch
```

Process large datasets of requirements.

---

### Analytics

```
GET /api/analytics_data
```

Returns classification metrics.

---

### Usage Metrics

```
GET /api/usage_data
```

Returns token usage and cost statistics.

---

### Batch Runs

```
GET /api/batch_runs
```

Returns history of batch experiments.

---

### Prompt Technique Comparison

```
GET /api/technique_comparison
```

Compare prompting strategies.

---

### Author

**Shashank Rajput**

GitHub:
https://github.com/ShashankRajput90

---

### License

Open source project. Add your preferred license.

---

### Acknowledgments

This project integrates multiple AI providers and open-source technologies including:

* Flask
* PostgreSQL
* Groq API
* Google Gemini
* Cohere
* Anthropic Claude
* Ollama (Mistral)

---

⭐ If you found this project useful, consider starring the repository!
