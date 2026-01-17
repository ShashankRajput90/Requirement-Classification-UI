import os
import re
import requests
from flask import Flask, render_template, request
from groq import Groq
import google.generativeai as genai
import cohere
from anthropic import Anthropic
from dotenv import load_dotenv

# =========================
# Load API Keys
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# =========================
# Helper: clean response
# =========================
def clean_response(raw_output: str) -> str:
    if not raw_output:
        return "⚠️ Empty response"

    # Remove hidden reasoning blocks if present
    raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    # Keep only lines starting with "1.", "2.", "3."
    lines = [line.strip() for line in raw_output.split("\n") if re.match(r"^[1-3]\.", line.strip())]

    if lines:
        return "\n".join(lines[:3])
    return raw_output.strip()

# =========================
# Prompting Techniques
# =========================
def build_prompt(user_story: str, technique: str) -> str:
    base_instruction = (
        "You are a software engineering assistant.\n"
        "Classify the user story as an NFR in exactly 3 lines:\n"
        "1. Is NFR: <Yes/No>\n"
        "2. NFR Type: <type>\n"
        "3. Reason: <short reason>\n"
        "No extra text, no hidden reasoning."
    )

    technique_prompts = {
        "zero_shot": f"{base_instruction}\n\nUser Story: \"{user_story}\"",

        "few_shot": f"""{base_instruction}

Examples:
- User Story: "The system shall be available 24/7."
  1. Is NFR: Yes
  2. NFR Type: Availability
  3. Reason: Continuous uptime required

- User Story: "The user can reset password using email."
  1. Is NFR: No
  2. NFR Type: -
  3. Reason: Functional requirement

Now classify:
User Story: \"{user_story}\"""",

        "chain_of_thought": f"""{base_instruction}
Think step by step internally, but output only the 3-line answer.
User Story: \"{user_story}\"""",

        "role_based": f"""You are an experienced software architect with 10+ years in requirements engineering.
{base_instruction}
User Story: \"{user_story}\"""",

        "react": f"""You will first reason internally, then act.
Output ONLY the 3-line answer.
{base_instruction}
User Story: \"{user_story}\""""
    }

    return technique_prompts.get(technique, technique_prompts["zero_shot"])

# =========================
# Model Functions
# =========================
def classify_with_groq_deepseek(user_story, technique):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = build_prompt(user_story, technique)
        response = client.chat.completions.create(
            # model="deepseek-r1-distill-llama-70b",
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return clean_response(response.choices[0].message.content)
    except Exception as e:
        return f"❌ Groq openai/gpt-oss-120b Error: {e}"

def classify_with_groq(user_story, technique):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        prompt = build_prompt(user_story, technique)
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        raw_output = response.json()["choices"][0]["message"]["content"]
        return clean_response(raw_output)
    except Exception as e:
        return f"❌ Groq Error: {e}"

def classify_with_gemini(user_story, technique):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = build_prompt(user_story, technique)
        response = model.generate_content(prompt)
        return clean_response(response.text)
    except Exception as e:
        return f"❌ Gemini Error: {e}"

def classify_with_cohere(user_story, technique):
    try:
        co = cohere.Client(COHERE_API_KEY)
        message = build_prompt(user_story, technique)
        response = co.chat(model="command-r-plus-08-2024", message=message)
        return clean_response(response.text)
    except Exception as e:
        return f"❌ Cohere Error: {e}"

def classify_with_claude(user_story, technique):
    try:
        client = Anthropic(api_key=CLAUDE_API_KEY)
        prompt = build_prompt(user_story, technique)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            temperature=0.2,
            system="You are a helpful assistant for requirements classification.",
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_response(response.content[0].text)
    except Exception as e:
        return f"❌ Claude Error: {e}"

def run_mistral_local(user_input, technique):
    try:
        MISTRAL_URL = "http://localhost:11434/api/generate"
        prompt = build_prompt(user_input, technique)
        response = requests.post(
            MISTRAL_URL,
            json={"model": "mistral:latest", "prompt": prompt, "stream": False}
        )
        return clean_response(response.json().get("response", ""))
    except Exception as e:
        return f"❌ Mistral Error: {e}"

# =========================
# Unified wrapper
# =========================
def classify(model_name, story, technique):
    if model_name == "groq_gpt": return classify_with_groq_deepseek(story, technique)
    if model_name == "groq_llama3": return classify_with_groq(story, technique)
    if model_name == "gemini": return classify_with_gemini(story, technique)
    if model_name == "cohere": return classify_with_cohere(story, technique)
    if model_name == "claude": return classify_with_claude(story, technique)
    if model_name == "mistral": return run_mistral_local(story, technique)
    return "❌ Unknown model"

# =========================
# Flask App
# =========================
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    story = ""
    model_choice = ""
    technique = "zero_shot"

    if request.method == "POST":
        story = request.form.get("story", "").strip()
        model_choice = request.form.get("model")
        technique = request.form.get("technique", "zero_shot")

        if story and model_choice:
            result = classify(model_choice, story, technique)

    return render_template("index.html", story=story, result=result,
                           model_choice=model_choice, technique=technique)

if __name__ == "__main__":
    app.run(debug=True)
