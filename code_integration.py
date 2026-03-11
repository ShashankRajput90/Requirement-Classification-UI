import os
import re
import requests
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

    # Capture <think> block if it exists
    think_match = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""

    # Remove <think> block to get the rest of the text for classification
    classification_text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    lines = [
        line.strip()
        for line in classification_text.split("\n")
        if re.match(r"^[1-4]\.", line.strip())
    ]

    final_text = ""
    if lines:
        final_text = "\n".join(lines[:4])
    else:
        final_text = classification_text.strip()
        
    if think_content:
        return f"<think>\n{think_content}\n</think>\n{final_text}"
    return final_text

# =========================
# Prompt Builder
# =========================
def build_prompt(user_story: str, technique: str) -> str:
    base_instruction = (
        "You are a software engineering assistant.\n"
        "First, output your internal reasoning process enclosed in `<think>` and `</think>` tags.\n"
        "In your reasoning, strictly follow these 5 steps (leave a blank line between each step):\n"
        "Step 1: Identify the main requirement in the user story.\n"
        "Step 2: Determine whether it describes functionality or quality.\n"
        "Step 3: Compare with NFR definitions.\n"
        "Step 4: Evaluate whether the story fits an NFR category.\n"
        "Step 5: Final reasoning conclusion.\n\n"
        "After the `</think>` tag, classify the user story as an NFR in exactly 4 lines, no extra text:\n"
        "1. Is NFR: <Yes/No>\n"
        "2. NFR Type: <type if NFR, else write 'N/A'>\n"
        "3. Reason: <short reason why it is or is not an NFR>\n"   # ← force reason even for FR
        "4. Confidence: <number between 0 and 100>\n"
        "Always fill every line. Never leave a line blank."        # ← explicit instruction
    )

    technique_prompts = {
        "zero_shot": f"{base_instruction}\n\nUser Story: \"{user_story}\"",

        "few_shot": f"""{base_instruction}

Examples:
- User Story: "The system shall be available 24/7."
  <think>
  Step 1: Identify the main requirement in the user story.
  The main requirement is that the system must have continuous 24/7 availability.

  Step 2: Determine whether it describes functionality or quality.
  This describes how well the system operates (quality), not a specific feature or user action.

  Step 3: Compare with NFR definitions.
  NFRs define system attributes such as performance, security, and availability.

  Step 4: Evaluate whether the story fits an NFR category.
  The constraint exactly matches the "Availability" category of NFRs.

  Step 5: Final reasoning conclusion.
  Since the story dictates an exact uptime requirement of 24/7 and describes a system property, it is an Availability NFR.
  </think>
  1. Is NFR: Yes
  2. NFR Type: Availability
  3. Reason: Specifies an uptime constraint, not a feature
  4. Confidence: 95

- User Story: "The user can reset password using email."
  <think>
  Step 1: Identify the main requirement in the user story.
  The user wants to be able to reset their password via email.

  Step 2: Determine whether it describes functionality or quality.
  This describes a specific action the user can take (a feature/functionality), rather than how the system performs.

  Step 3: Compare with NFR definitions.
  NFRs are about system qualities (e.g., speed, security); functional requirements are about what the system does.

  Step 4: Evaluate whether the story fits an NFR category.
  This does not fit any NFR category like performance or security; it's a standard user capability.

  Step 5: Final reasoning conclusion.
  Because this story describes a specific functional feature of the system, it is a Functional Requirement (not an NFR).
  </think>
  1. Is NFR: No
  2. NFR Type: N/A
  3. Reason: Describes a user-facing feature, not a quality attribute
  4. Confidence: 90

Now classify:
User Story: "{user_story}"
""",

        "chain_of_thought": f"""You are a software engineering assistant.
Think step by step in `<think>...</think>` tags to decide if the user story is a Non-Functional Requirement (NFR).
In your reasoning, strictly follow these 5 steps:
Step 1: Identify the main requirement in the user story.
Step 2: Determine whether it describes functionality or quality.
Step 3: Compare with NFR definitions.
Step 4: Evaluate whether the story fits an NFR category.
Step 5: Final reasoning conclusion.

Then output ONLY the following 4 lines after the think tag:

1. Is NFR: <Yes/No>
2. NFR Type: <type if NFR, else N/A>
3. Reason: <one sentence>
4. Confidence: <0-100>

User Story: "{user_story}"
""",

        "role_based": f"""You are an experienced software architect specializing in requirements engineering.
Classify the following user story. Output your reasoning in `<think>...</think>` tags first.
In your reasoning, strictly follow these 5 steps:
Step 1: Identify the main requirement in the user story.
Step 2: Determine whether it describes functionality or quality.
Step 3: Compare with NFR definitions.
Step 4: Evaluate whether the story fits an NFR category.
Step 5: Final reasoning conclusion.

Then respond in exactly 4 lines, always filling every field:
1. Is NFR: <Yes/No>
2. NFR Type: <type if NFR, else N/A>
3. Reason: <one sentence explaining your decision>
4. Confidence: <0-100>

User Story: "{user_story}"
""",

        "react": f"""Reason internally in `<think>...</think>` tags and output only the final answer afterwards.
In your reasoning, strictly follow these 5 steps:
Step 1: Identify the main requirement in the user story.
Step 2: Determine whether it describes functionality or quality.
Step 3: Compare with NFR definitions.
Step 4: Evaluate whether the story fits an NFR category.
Step 5: Final reasoning conclusion.

Classify the user story in exactly 4 lines, always filling every field:
1. Is NFR: <Yes/No>
2. NFR Type: <type if NFR, else N/A>
3. Reason: <one sentence>
4. Confidence: <0-100>

User Story: "{user_story}"
"""
    }

    return technique_prompts.get(technique, technique_prompts["zero_shot"])

def handle_llm_exception(provider_name: str, error: Exception):
    error_message = str(error).lower()
    
    if (
        "winerror 10061" in error_message
        or "failed to establish a new connection" in error_message
        or "connection refused" in error_message
        or "httpconnectionpool" in error_message
    ):
        return {
            "error": f"{provider_name} service is not running. "
                     f"If using Mistral locally, start Ollama with: 'ollama serve'"
        }, 503

    # Rate limit detection
    if "429" in error_message or "rate limit" in error_message:
        return {
            "error": f"{provider_name} API rate limit exceeded. Please try again later."
        }, 429

    # Quota exceeded
    if "quota" in error_message or "resource_exhausted" in error_message:
        return {
            "error": f"{provider_name} API quota exceeded. Please upgrade your plan or try later."
        }, 429

    # Authentication error
    if "401" in error_message or "unauthorized" in error_message:
        return {
            "error": f"{provider_name} API key is invalid or missing."
        }, 401

    # Model not found
    if "model" in error_message and "not found" in error_message:
        return {
            "error": f"{provider_name} model not available."
        }, 400

    # Default
    return {
        "error": f"{provider_name} Error: {str(error)}"
    }, 500

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
        return result, 200

    except Exception as e:
        return handle_llm_exception("Groq GPT-OSS", e)

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
        return clean_response(raw_output), 200

    except Exception as e:
        return handle_llm_exception("Groq LLaMA3", e)

def classify_with_gemini(user_story, technique):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = build_prompt(user_story, technique)
        response = model.generate_content(prompt)

        return clean_response(response.text), 200

    except Exception as e:
        return handle_llm_exception("Gemini", e)

def classify_with_cohere(user_story, technique):
    try:
        co = cohere.Client(COHERE_API_KEY)
        prompt = build_prompt(user_story, technique)

        response = co.chat(
            model="command-r-plus-08-2024",
            message=prompt
        )

        return clean_response(response.text), 200

    except Exception as e:
        return handle_llm_exception("Cohere", e)
    
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

        return clean_response(response.content[0].text), 200

    except Exception as e:
        return handle_llm_exception("Claude", e)

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

        return clean_response(data.get("response", "")), 200

    except Exception as e:
        return handle_llm_exception("Mistral", e)

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

    return {"error": "Unknown model"}, 500