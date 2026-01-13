import requests
import time
import httpx
import re
import joblib
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from google import genai
import cohere
from anthropic import Anthropic
from typing import List, Literal, Dict, Any

# Import settings from the new config file
from config import settings

# =========================
# Initialize LLM Clients
# =========================
try:
    if settings.GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    if settings.COHERE_API_KEY:
        co = cohere.Client(settings.COHERE_API_KEY)
    if settings.CLAUDE_API_KEY:
        claude_client = Anthropic(api_key=settings.CLAUDE_API_KEY)
    if settings.GROQ_API_KEY:
        groq_client = Groq(api_key=settings.GROQ_API_KEY)
except Exception as e:
    print(f"Warning: Could not initialize one or more API clients. {e}")

# =========================
# Load Local ML Models
# =========================
try:
    tfidf_vectorizer = joblib.load(settings.TFIDF_VECTORIZER_PATH)
    logistic_model = joblib.load(settings.LOGISTIC_MODEL_PATH)
    bert_tokenizer = joblib.load(settings.BERT_TOKENIZER_PATH)
    bert_model = joblib.load(settings.BERT_MODEL_PATH)
    print("All .pkl models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading .pkl model: {e}")
    print("Please check the file paths in your .env or config.py file.")
    tfidf_vectorizer = logistic_model = bert_tokenizer = bert_model = None

# =========================
# FastAPI App Setup
# =========================
app = FastAPI(
    title="Unified Requirement Classifier API",
    description="One API for both local ML classification and external LLM comparison."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (e.g., opening index.html from file://)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API Schemas (Pydantic Models)
# =========================

# --- Schemas for Local Classifier ---
class UserStory(BaseModel):
    user_story: str

class ClassificationResponse(BaseModel):
    classification: str
    reason: str

class BatchRequest(BaseModel):
    stories: List[str]

class BatchResponse(BaseModel):
    original_story: str
    classification: str
    reason: str

class StoryWithClassification(UserStory):
    classification: str

class GeminiResponse(BaseModel):
    generated_text: str

# --- Schemas for LLM Comparator ---
ModelChoice = Literal[
    "groq_llama3", "groq_deepseek", "gemini", 
    "cohere", "claude", "mistral_local"
]
TechniqueChoice = Literal[
    "zero_shot", "few_shot", "chain_of_thought", "role_based", "react"
]

class LLMClassifierRequest(BaseModel):
    story: str
    model: ModelChoice
    technique: TechniqueChoice

class LLMClassifierResponse(BaseModel):
    result: str
    model: ModelChoice
    technique: TechniqueChoice

# ===============================================
# SECTION 1: Local Model Classification Logic
# ===============================================

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def _classify_story(story_text: str) -> ClassificationResponse:
    """Internal function to run the local classification models."""
    if not all([tfidf_vectorizer, logistic_model, bert_tokenizer, bert_model]):
        raise HTTPException(status_code=503, detail="Local ML models are not loaded. Check server logs.")
        
    # Part 1: FR/NFR Classification (Logistic Regression)
    processed_story = preprocess_text(story_text)
    story_tfidf = tfidf_vectorizer.transform([processed_story])
    prediction = logistic_model.predict(story_tfidf)[0]

    # Part 2: Reason Generation (BERT)
    if prediction == 0:
        return ClassificationResponse(
            classification="Non-Functional Requirement",
            reason="N/A (This is a non-functional requirement)"
        )
    else:
        # Load reason_map from settings
        reason_map = settings.BERT_REASON_MAP
        
        inputs = bert_tokenizer(
            story_text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
        reason_index = torch.argmax(logits, dim=1).item()
        reason_text = reason_map.get(reason_index, f"Undefined Reason (Class Index: {reason_index})")

        return ClassificationResponse(
            classification="Functional Requirement",
            reason=reason_text
        )

@app.post("/classify", response_model=ClassificationResponse)
def classify_requirement_endpoint(story: UserStory):
    """Classifies a *single* user story using the local TF-IDF/BERT models."""
    try:
        return _classify_story(story.user_story)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-classify", response_model=List[BatchResponse])
def batch_classify_endpoint(request: BatchRequest):
    """Classifies a *batch* of user stories using the local TF-IDF/BERT models."""
    results = []
    for story in request.stories:
        try:
            classification_result = _classify_story(story)
            results.append(BatchResponse(
                original_story=story,
                classification=classification_result.classification,
                reason=classification_result.reason
            ))
        except Exception as e:
            results.append(BatchResponse(
                original_story=story,
                classification="ERROR",
                reason=str(e)
            ))
    return results

# ===============================================
# SECTION 2: Gemini Helper Logic
# ===============================================

async def call_gemini_api_internal(system_prompt: str, user_prompt: str) -> str:
    """Internal helper for all Gemini calls."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not configured.")
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={settings.GEMINI_API_KEY}",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
            if not text:
                raise ValueError("No text returned from Gemini API.")
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling Gemini: {e}")

@app.post("/improve-story", response_model=GeminiResponse)
async def improve_story_endpoint(story: UserStory):
    """Uses Gemini to rewrite a rough idea into a formal user story."""
    system_prompt = (
        "You are an expert Agile Product Manager. "
        "Rewrite the user's idea into a single, formal user story: "
        "'As a [type of user], I want [some goal] so that [some reason]'. "
        "Only return the single user story."
    )
    generated_text = await call_gemini_api_internal(system_prompt, story.user_story)
    return GeminiResponse(generated_text=generated_text.strip())

@app.post("/suggest-ac", response_model=GeminiResponse)
async def suggest_ac_endpoint(data: StoryWithClassification):
    """Uses Gemini to generate Acceptance Criteria for a user story."""
    system_prompt = (
        "You are an expert QA Engineer. "
        "Generate 3-5 clear, bulleted acceptance criteria (AC) for the given user story. "
        "Only return the bulleted list."
    )
    user_prompt = f"User Story: \"{data.user_story}\"\nClassification: {data.classification}\n\nGenerate 3-5 acceptance criteria:"
    generated_text = await call_gemini_api_internal(system_prompt, user_prompt)
    return GeminiResponse(generated_text=generated_text.strip())

# ===============================================
# SECTION 3: External LLM Comparator Logic
# ===============================================

def clean_llm_response(raw_output: str) -> str:
    """Cleans raw LLM output to the 3-line format."""
    # (Copied from your code_integration.py)
    if not raw_output:
        return "⚠️ Empty response"
    if "<think>" in raw_output:
        raw_output = raw_output.split("</think>")[-1].strip()
    lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
    
    formatted_lines = []
    other_lines = []
    for line in lines:
        if re.match(r"^\d\.", line) or "Is NFR" in line or "NFR Type" in line or "Primary Reason" in line:
            formatted_lines.append(line)
        else:
            other_lines.append(line)
    
    final_lines = formatted_lines + [l for l in other_lines if l not in formatted_lines]

    if len(final_lines) >= 3:
        return "\n".join(final_lines[:3])
    elif final_lines:
        return "\n".join(final_lines)
    else:
        return raw_output.strip()

def build_prompt(user_story: str, technique: str) -> str:
    """Builds a prompt based on the selected technique."""
    base_instruction = """
You are a software engineering assistant.
Respond ONLY in exactly 3 lines in this format:
1. Is NFR: <Yes/No>
2. NFR Type: <type>
3. Primary Reason: <reason>
"""
    if technique == "zero_shot":
        return f"{base_instruction}\nUser Story: \"{user_story}\""
    elif technique == "few_shot":
        return f"""
{base_instruction}
User Story: "As a user, I want the system to load the dashboard in under 3 seconds."
1. Is NFR: Yes
2. NFR Type: Performance
3. Primary Reason: Sets a specific, measurable time constraint for a system operation.

User Story: "As a user, I want to log in with my email and password."
1. Is NFR: No
2. NFR Type: N/A
3. Primary Reason: Describes a specific function (authentication) the user performs.

User Story: "{user_story}"
"""
    elif technique == "chain_of_thought":
        return f"""
{base_instruction}
Let's think step by step.
1. Analyze the user story: "{user_story}"
2. Does it describe a function (what the system does) or a quality (how it does it)?
3. If it's a quality, what kind of quality is it (e.g., Performance, Security, Usability)?
4. Formulate the 3-line answer.

User Story: "{user_story}"
"""
    # (Add your other techniques here)
    else: # Default
        return f"{base_instruction}\nUser Story: \"{user_story}\""

# --- LLM Classifier Functions ---
def classify_with_groq(story: str, technique: str, model_name: str) -> str:
    prompt = build_prompt(story, technique)
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model_name
        )
        return clean_llm_response(chat_completion.choices[0].message.content)
    except Exception as e:
        return f"❌ Groq ({model_name}) Error: {e}"

def classify_with_gemini_llm(story: str, technique: str) -> str:
    prompt = build_prompt(story, technique)
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        return clean_llm_response(response.text)
    except Exception as e:
        return f"❌ Gemini Error: {e}"

def classify_with_cohere_llm(story: str, technique: str) -> str:
    prompt = build_prompt(story, technique)
    try:
        response = co.chat(message=prompt)
        return clean_llm_response(response.text)
    except Exception as e:
        return f"❌ Cohere Error: {e}"

def classify_with_claude_llm(story: str, technique: str) -> str:
    prompt = build_prompt(story, technique)
    try:
        message = claude_client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_llm_response(message.content[0].text)
    except Exception as e:
        return f"❌ Claude Error: {e}"

def classify_with_mistral_local(story: str, technique: str) -> str:
    prompt = build_prompt(story, technique)
    payload = {"model": "mistral", "prompt": prompt, "stream": False}
    try:
        response = requests.post(settings.MISTRAL_URL, json=payload)
        response.raise_for_status()
        return clean_llm_response(response.json().get("response", ""))
    except requests.exceptions.ConnectionError:
        return f"❌ Mistral Error: Could not connect to {settings.MISTRAL_URL}. Is the local server running?"
    except Exception as e:
        return f"❌ Mistral Error: {e}"


@app.get('/model-health')
def model_health():
    """Probe the configured Mistral URL and return diagnostic information.

    Useful for debugging connection refused / timeout errors from the UI.
    """
    mistral_url = getattr(settings, 'MISTRAL_URL', None)
    if not mistral_url:
        return {"reachable": False, "error": "MISTRAL_URL not configured in settings."}

    payload = {"model": "mistral:latest", "prompt": "Ping", "stream": False}
    timeout = getattr(settings, 'MISTRAL_TIMEOUT', 5)
    try:
        start = time.time()
        resp = requests.post(mistral_url, json=payload, timeout=timeout)
        elapsed = time.time() - start
        try:
            raw = resp.json()
        except Exception:
            raw = resp.text
        return {
            "reachable": resp.ok,
            "status_code": resp.status_code,
            "elapsed": elapsed,
            "raw": (raw if isinstance(raw, dict) else str(raw)[:200]),
            "error": None if resp.ok else f"HTTP {resp.status_code}"
        }
    except requests.exceptions.ConnectTimeout as e:
        return {"reachable": False, "error": f"ConnectTimeout: {str(e)}"}
    except requests.exceptions.ConnectionError as e:
        return {"reachable": False, "error": f"ConnectionError: {str(e)}"}
    except requests.exceptions.ReadTimeout as e:
        return {"reachable": False, "error": f"ReadTimeout after {timeout}s: {str(e)}"}
    except Exception as e:
        return {"reachable": False, "error": str(e)}

@app.post("/classify-llm", response_model=LLMClassifierResponse)
def classify_llm_endpoint(request: LLMClassifierRequest):
    """Classifies a user story using the selected *external* LLM and prompt technique."""
    story = request.story
    technique = request.technique
    model_choice = request.model
    result = ""

    try:
        if model_choice == "groq_llama3":
            result = classify_with_groq(story, technique, "llama3-8b-8192")
        elif model_choice == "groq_deepseek":
            result = classify_with_groq(story, technique, "deepseek-coder-33b-instruct") # Example model
        elif model_choice == "gemini":
            result = classify_with_gemini_llm(story, technique)
        elif model_choice == "cohere":
            result = classify_with_cohere_llm(story, technique)
        elif model_choice == "claude":
            result = classify_with_claude_llm(story, technique)
        elif model_choice == "mistral_local":
            result = classify_with_mistral_local(story, technique)
        else:
            raise HTTPException(status_code=400, detail="Invalid model choice")

        return LLMClassifierResponse(
            result=result, model=model_choice, technique=technique
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================================
# Run the Application
# ===============================================
if __name__ == "__main__":
    import uvicorn
    print("Starting Unified FastAPI server on http://127.0.0.1:8000")
    print("Loading settings from .env and config.py...")
    print(f"Local model paths: {settings.TFIDF_VECTORIZER_PATH}, {settings.LOGISTIC_MODEL_PATH}, etc.")
    print("Ensure .env file is present and local ML models (.pkl) are in this directory.")
    uvicorn.run(app, host="127.0.0.1", port=8000)