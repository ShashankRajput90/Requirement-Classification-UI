# Requirement Classification UI

A comprehensive web application for classifying software requirements (user stories) as **Functional Requirements (FR)** or **Non-Functional Requirements (NFR)** using multiple AI models and prompt engineering techniques.

## 🌟 Features

### Multi-Model Support
- **Gemini 2.5 Pro** - Google's latest generative AI model
- **Groq LLaMA 3.3** - Fast inference with LLaMA architecture
- **Groq DeepSeek** - Advanced code understanding
- **Cohere Command R+** - Enterprise-grade language model
- **Claude 3 Haiku** - Anthropic's efficient model
- **Mistral Local** - Self-hosted local inference

### Classification Capabilities
- **Single Classification** - Classify individual user stories in real-time
- **Batch Processing** - Upload CSV files for bulk classification
- **Model Comparison** - Compare performance across different models and prompts
- **Analytics Dashboard** - Visualize classification metrics and trends

### Prompt Engineering Techniques
- **Zero Shot** - Direct classification without examples
- **Few Shot** - Classification with example demonstrations
- **Chain of Thought** - Step-by-step reasoning approach

### Local ML Models
- TF-IDF + Logistic Regression for FR/NFR classification
- BERT-based model for reason generation
- Pre-trained model support for offline inference

## 📋 Requirements

### Python Dependencies
```
anthropic
cohere
fastapi
flask
google-generativeai
groq
httpx
joblib
matplotlib
numpy
pandas
plotly
pydantic
python-dotenv
requests
scikit-learn
torch
tqdm
transformers
uvicorn
werkzeug
```

### API Keys Required
You'll need API keys for the LLM services you want to use:
- `GEMINI_API_KEY` - Google AI Studio
- `GROQ_API_KEY` - Groq Cloud
- `COHERE_API_KEY` - Cohere Platform
- `CLAUDE_API_KEY` - Anthropic Console

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/ShashankRajput90/Requirement-Classification-UI.git
cd Requirement-Classification-UI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
CLAUDE_API_KEY=your_claude_key_here

# Local model paths
TFIDF_VECTORIZER_PATH=./models/tfidf_vectorizer. pkl
LOGISTIC_MODEL_PATH=./models/logistic_model.pkl
BERT_TOKENIZER_PATH=./models/bert_tokenizer.pkl
BERT_MODEL_PATH=./models/bert_model.pkl

# Mistral local server (optional)
MISTRAL_URL=http://localhost:11434/api/generate
```

4. **Download/Train ML Models** (if using local models)
Ensure your trained models are in the specified paths or train new ones using your dataset.

## 💻 Usage

### Running the Flask Application

```bash
python flask_app.py
```

The application will be available at `http://localhost:5000`

### Running the FastAPI Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

### Using the Web Interface

1. **Single Classification**
   - Navigate to the Single Classification page
   - Select your preferred model and prompt technique
   - Enter a user story
   - Click "Classify" to get results

2. **Batch Processing**
   - Go to Batch Processing page
   - Upload a CSV file with a `user_story` or `story` column
   - Select model and prompt settings
   - Process and download results

3. **Model Comparison**
   - Compare multiple models side-by-side
   - Test with sample stories or custom input
   - View performance metrics

4. **Analytics Dashboard**
   - View classification history
   - Analyze model performance
   - Download historical data

## 🔌 API Endpoints

### FastAPI Endpoints (`main.py`)

#### Single Classification
```bash
POST /classify
Body: {"user_story": "As a user, I want to... "}
```

#### Batch Classification
```bash
POST /batch-classify
Body: {"stories": ["story1", "story2", ...]}
```

#### LLM Classification
```bash
POST /classify-llm
Body: {
  "story": "user story text",
  "model": "gemini",
  "technique": "few_shot"
}
```

#### Improve Story with AI
```bash
POST /improve-story
Body: {"user_story": "rough idea"}
```

#### Generate Acceptance Criteria
```bash
POST /suggest-ac
Body: {
  "user_story": "story text",
  "classification": "Functional Requirement"
}
```

### Flask API Endpoints (`flask_app.py`)

- `POST /api/classify` - Single classification
- `POST /api/batch_classify` - Batch processing
- `POST /api/compare_models` - Model comparison
- `GET /api/analytics_data` - Analytics data
- `GET /api/download_history` - Export history
- `POST /api/clear_history` - Clear session data

## 📊 Sample User Stories

**Functional Requirement Examples:**
- "As a user, I want to log into the system using my email and password"
- "As a user, I want to search for products by name"

**Non-Functional Requirement Examples:**
- "As a user, I want the system to respond within 2 seconds" (Performance)
- "As a user, I want my data to be encrypted and secure" (Security)

## 🏗️ Project Structure

```
Requirement-Classification-UI/
├── flask_app.py              # Flask web application
├── main.py                   # FastAPI application
├── config.py                 # Configuration settings
├── code_integration.py       # LLM integration functions
├── evaluation. py             # Evaluation metrics
├── batch_evaluation.py       # Batch evaluation utilities
├── Batches. py               # Batch processing helpers
├── interactive_flask. py     # Interactive Flask features
├── requirements.txt         # Python dependencies
├── templates/               # HTML templates
│   └── index.html
└── utils/                   # Utility functions
```

## 🔧 Configuration

Edit `config.py` to customize:
- Model paths
- API endpoints
- Classification thresholds
- BERT reason mappings
- Timeout settings

## 📈 Model Performance

The application provides metrics including:
- Classification accuracy (FR vs NFR)
- Average processing time
- Model error rates
- Confidence scores
- NFR type distribution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source.  Please add your preferred license.

## 👨‍💻 Author

**Shashank Rajput**
- GitHub: [@ShashankRajput90](https://github.com/ShashankRajput90)

## 🙏 Acknowledgments

- Built with Flask and FastAPI
- Powered by Transformers and Scikit-learn
- Integrated with leading LLM providers
- UI components using modern web standards

## 📞 Support

For issues and questions: 
- Open an issue on GitHub
- Check existing documentation
- Review API documentation at `/docs` endpoint

---

**Note:** Ensure you have valid API keys and trained models before running the application. Some features require internet connectivity for cloud-based LLM services. 
