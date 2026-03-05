import os
import re
import requests
from groq import Groq
# import google.generativeai as genai
from google import genai
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

    raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    lines = [
        line.strip()
        for line in raw_output.split("\n")
        if re.match(r"^[1-4]\.", line.strip())
    ]

    if lines:
        return "\n".join(lines[:4])
    return raw_output.strip()

# =========================
# Prompt Builder
# =========================
def build_prompt(user_story: str, technique: str) -> str:
    base_instruction = (
        "You are a software engineering assistant.\n"
        "Classify the user story as an NFR in exactly 4 lines, no extra text:\n"
        "1. Is NFR: <Yes/No>\n"
        "2. NFR Type: <type if NFR, else write 'N/A'>\n"
        "3. Reason: <short reason why it is or is not an NFR>\n"
        "4. Confidence: <number between 0 and 100>\n"
        "Always fill every line. Never leave a line blank."
    )

    technique_prompts = {
        "zero_shot": f"{base_instruction}\n\nUser Story: \"{user_story}\"",

        "few_shot": f"""{base_instruction}

Examples:
- User Story: "The system shall be available 24/7."
  1. Is NFR: Yes
  2. NFR Type: Availability
  3. Reason: Specifies an uptime constraint, not a feature
  4. Confidence: 95

- User Story: "The user can reset password using email."
  1. Is NFR: No
  2. NFR Type: N/A
  3. Reason: Describes a user-facing feature, not a quality attribute
  4. Confidence: 90

Now classify:
User Story: "{user_story}"
""",

        "chain_of_thought": f"""You are a software engineering assistant.
Think step by step to decide if the user story is a Non-Functional Requirement (NFR).
Then output ONLY the following 4 lines, nothing else:

1. Is NFR: <Yes/No>
2. NFR Type: <type if NFR, else N/A>
3. Reason: <one sentence>
4. Confidence: <0-100>

User Story: "{user_story}"
""",

        "role_based": f"""You are an experienced software architect specializing in requirements engineering.
Classify the following user story. Respond in exactly 4 lines, always filling every field:
1. Is NFR: <Yes/No>
2. NFR Type: <type if NFR, else N/A>
3. Reason: <one sentence explaining your decision>
4. Confidence: <0-100>

User Story: "{user_story}"
""",

        "react": f"""Reason internally and output only the final answer.
Classify the user story in exactly 4 lines, always filling every field:
1. Is NFR: <Yes/No>
2. NFR Type: <type if NFR, else N/A>
3. Reason: <one sentence>
4. Confidence: <0-100>

User Story: "{user_story}"
"""
    }

    return technique_prompts.get(technique, technique_prompts["zero_shot"])
    
# =========================
# MODEL FUNCTIONS
# =========================
def classify_with_groq_deepseek(user_story, technique):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = build_prompt(user_story, technique)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        result = clean_response(response.choices[0].message.content)
        usage = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens
        }
        return result, 200, usage

    except Exception as e:
        return {"error": f"Groq Error: {str(e)}"}, 500, {"prompt": 0, "completion": 0}


def classify_with_groq(user_story, technique):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

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
        usage_data = response.json().get("usage", {})
        usage = {
            "prompt": usage_data.get("prompt_tokens", 0),
            "completion": usage_data.get("completion_tokens", 0)
        }
        return clean_response(raw_output), 200, usage

    except Exception as e:
        return {"error": f"Groq LLaMA Error: {str(e)}"}, 500, {"prompt": 0, "completion": 0}


# def classify_with_gemini(user_story, technique):
#     try:
#         genai.configure(api_key=GEMINI_API_KEY)
#         model = genai.GenerativeModel("models/gemini-2.5-pro")
#         prompt = build_prompt(user_story, technique)
#         response = model.generate_content(prompt)

#         return clean_response(response.text), 200

#     except Exception as e:
#         return {"error": f"Gemini Error: {str(e)}"}, 500
def classify_with_gemini(user_story, technique):
    try:
        # ✅ Create a client with your API key
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build the prompt
        prompt = build_prompt(user_story, technique)

        # Generate content using the new SDK
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )

        # Extract text from response
        # response is a dict-like object: response.output[0].content[0].text
        raw_output = response.output[0].content[0].text
        
        # Try to parse usage_metadata if available in the new SDK
        prompt_tokens = 0
        completion_tokens = 0
        try:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
        except AttributeError:
            pass
            
        return clean_response(raw_output), 200, {"prompt": prompt_tokens, "completion": completion_tokens}

    except Exception as e:
        error_message = str(e)

        # Check if it's a quota/limit error
        if "RESOURCE_EXHAUSTED" in error_message or "quota exceeded" in error_message.lower():
            # Friendly message for frontend
            return {"error": "Gemini API limit exceeded. Please check your quota or try again later."}, 429, {"prompt": 0, "completion": 0}
        else:
            return {"error": f"Gemini Error: {error_message}"}, 500, {"prompt": 0, "completion": 0}


def classify_with_cohere(user_story, technique):
    try:
        co = cohere.Client(COHERE_API_KEY)
        prompt = build_prompt(user_story, technique)

        response = co.chat(
            model="command-r-plus-08-2024",
            message=prompt
        )

        prompt_tokens = 0
        completion_tokens = 0
        if response.meta and response.meta.billed_units:
            prompt_tokens = response.meta.billed_units.input_tokens
            completion_tokens = response.meta.billed_units.output_tokens

        return clean_response(response.text), 200, {"prompt": prompt_tokens, "completion": completion_tokens}

    except Exception as e:
        error_message = str(e)
        if "429" in error_message or "trial key" in error_message.lower() or "rate limit" in error_message.lower():
            return {"error": "Cohere API limit exceeded. Your Trial key is limited to 1000 calls/month. Upgrade at https://dashboard.cohere.com/api-keys"}, 429, {"prompt": 0, "completion": 0}
        return {"error": f"Cohere Error: {error_message}"}, 500, {"prompt": 0, "completion": 0}


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

        usage = {"prompt": 0, "completion": 0}
        if hasattr(response, 'usage'):
            usage["prompt"] = getattr(response.usage, 'input_tokens', 0)
            usage["completion"] = getattr(response.usage, 'output_tokens', 0)

        return clean_response(response.content[0].text), 200, usage

    except Exception as e:
        return {"error": f"Claude Error: {str(e)}"}, 500, {"prompt": 0, "completion": 0}


def run_mistral_local(user_story, technique):
    try:
        MISTRAL_URL = "http://localhost:11434/api/generate"
        prompt = build_prompt(user_story, technique)

        response = requests.post(
            MISTRAL_URL,
            json={
                "model": "mistral:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )

        response.raise_for_status()
        data = response.json()
        
        usage = {
            "prompt": data.get("prompt_eval_count", 0),
            "completion": data.get("eval_count", 0)
        }

        return clean_response(data.get("response", "")), 200, usage

    except requests.exceptions.ConnectionError:
        return {"error": "Mistral is not running. Please start Ollama locally by running: ollama serve"}, 503, {"prompt": 0, "completion": 0}

    except requests.exceptions.Timeout:
        return {"error": "Mistral timed out. The model may be loading, please try again."}, 504, {"prompt": 0, "completion": 0}

    except Exception as e:
        return {"error": f"Mistral Error: {str(e)}"}, 500, {"prompt": 0, "completion": 0}


# =========================
# UNIFIED WRAPPER
# =========================
def classify(model_name, story, technique):

    if model_name == "groq_gpt":
        return classify_with_groq_deepseek(story, technique)

    if model_name == "groq_llama3":
        return classify_with_groq(story, technique)

    if model_name == "gemini":
        return classify_with_gemini(story, technique)

    if model_name == "cohere":
        return classify_with_cohere(story, technique)

    if model_name == "claude":
        return classify_with_claude(story, technique)

    if model_name == "mistral":
        return run_mistral_local(story, technique)

    return {"error": "Unknown model"}, 500, {"prompt": 0, "completion": 0}