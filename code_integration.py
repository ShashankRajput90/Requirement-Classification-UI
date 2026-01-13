import os
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
# Helper: clean and enforce 3 lines
# =========================
def clean_response(raw_output: str) -> str:
    if not raw_output:
        return "⚠️ Empty response"

    if "<think>" in raw_output:  # strip hidden thoughts
        raw_output = raw_output.split("</think>")[-1].strip()

    lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
    if len(lines) >= 3:
        return "\n".join(lines[:3])
    else:
        return raw_output.strip()

# =========================
# Prompting Techniques
# =========================
def build_prompt(user_story: str, technique: str) -> str:
    base_instruction = """
You are a software engineering assistant.
Respond ONLY in exactly 3 lines in this format:
1. Is NFR: <Yes/No>
2. NFR Type: <type>
3. Reason: <one line reason>
Do not include reasoning steps or extra text.
"""

    if technique == "zero_shot":
        return f"{base_instruction}\n\nUser Story: \"{user_story}\""

    elif technique == "few_shot":
        examples = """
Example 1:
User Story: "The system shall be available 24/7."
1. Is NFR: Yes
2. NFR Type: Availability
3. Reason: Requires continuous uptime

Example 2:
User Story: "The user can reset password using email."
1. Is NFR: No
2. NFR Type: -
3. Reason: Functional requirement
"""
        return f"{base_instruction}\n{examples}\nClassify now:\nUser Story: \"{user_story}\""

    elif technique == "chain_of_thought":
        return f"""{base_instruction}
Think step by step (but only show the final 3 lines, not your reasoning).
User Story: "{user_story}"
"""

    elif technique == "role_based":
        return f"""
You are an experienced software architect with 10+ years in requirements engineering.
{base_instruction}
User Story: "{user_story}"
"""

    elif technique == "react":
        return f"""
You will reason and act in two phases:
- Reason: Think briefly about classification
- Act: Output only the final 3-line classification

{base_instruction}
User Story: "{user_story}"
"""

    else:  # default
        return f"{base_instruction}\n\nUser Story: \"{user_story}\""

# =========================
# Prompt Builder (EXTRACTED FROM main.py)
# =========================

class PromptBuilder:
    BASE_INSTRUCTION = """
You are a software engineering assistant.
Respond ONLY in exactly 3 lines:
1. Is NFR: <Yes/No>
2. NFR Type: <type>
3. Reason: <one line>
"""

    @staticmethod
    def build(story: str, technique: str = "zero_shot") -> str:
        if technique == "zero_shot":
            return f"{PromptBuilder.BASE_INSTRUCTION}\nUser Story: \"{story}\""

        if technique == "few_shot":
            return f"""
{PromptBuilder.BASE_INSTRUCTION}
User Story: "The system must respond in under 2 seconds."
1. Is NFR: Yes
2. NFR Type: Performance
3. Reason: Response time constraint

User Story: "{story}"
"""

        return f"{PromptBuilder.BASE_INSTRUCTION}\nUser Story: \"{story}\""

class ResponseCleaner:
    @staticmethod
    def clean(raw: str) -> str:
        if not raw:
            return "⚠️ Empty response"

        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return "\n".join(lines[:3])

# =========================
# Model Functions
# =========================
def classify_with_groq_deepseek(user_story, technique):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = build_prompt(user_story, technique)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.0
        )
        return clean_response(response.choices[0].message.content)
    except Exception as e:
        return f"❌ Groq DeepSeek Error: {e}"

def classify_with_groq(user_story, technique):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        prompt = build_prompt(user_story, technique)
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": prompt}],
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
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = build_prompt(user_story, technique)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
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

def classify_with_mistral_local(user_input, technique):
    try:
        MISTRAL_URL = os.getenv('MISTRAL_URL', 'http://localhost:11434/api/generate')
        mistral_timeout = int(os.getenv('MISTRAL_TIMEOUT', '60'))
        prompt = build_prompt(user_input, technique)

        response = requests.post(
            MISTRAL_URL,
            json={"model": "mistral:latest", "prompt": prompt, "stream": False},
            timeout=mistral_timeout
        )
        response.raise_for_status()
        return clean_response(response.json().get("response", ""))
    except Exception as e:
        # Provide actionable messages
        import requests as _requests
        extra = ''
        if isinstance(e, _requests.exceptions.ReadTimeout):
            extra = f" Read timed out after {mistral_timeout}s. Try increasing MISTRAL_TIMEOUT or ensure the server is healthy."
        elif isinstance(e, (_requests.exceptions.ConnectionError, _requests.exceptions.ConnectTimeout)):
            extra = f" Could not connect to {MISTRAL_URL}. Ensure the server is running and MISTRAL_URL is correct."
        return f"❌ Mistral Error: {e}.{extra}"

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
            if model_choice == "groq_deepseek":
                result = classify_with_groq_deepseek(story, technique)
            elif model_choice == "groq_llama3":
                result = classify_with_groq(story, technique)
            elif model_choice == "gemini":
                result = classify_with_gemini(story, technique)
            elif model_choice == "cohere":
                result = classify_with_cohere(story, technique)
            elif model_choice == "claude":
                result = classify_with_claude(story, technique)
            elif model_choice == "mistral":
                result = classify_with_mistral_local(story, technique)

    return render_template("index.html", story=story, result=result,
                           model_choice=model_choice, technique=technique)

if __name__ == "__main__":
    app.run(debug=True)
