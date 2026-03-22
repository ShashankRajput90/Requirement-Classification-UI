"""
context_utils.py
----------------
Shared utilities for domain-context-aware classification and RQI scoring.
Used by:
  - app.py (Flask routes)
  - context_based_evaluation.py (offline evaluation script)
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# DOMAIN CONTEXT TEMPLATES
# ==========================
DOMAIN_CONTEXTS = {
    "Software Application": {
        "system": "Software system handling user accounts and transactions",
        "stakeholders": "End users, developers, administrators"
    },
    "Healthcare System": {
        "system": "Electronic health record system storing patient data",
        "stakeholders": "Doctors, nurses, hospital admins, patients"
    },
    "E-Commerce": {
        "system": "Online shopping platform managing products and orders",
        "stakeholders": "Customers, sellers, admins"
    },
    "Banking System": {
        "system": "Banking system handling financial transactions",
        "stakeholders": "Customers, bank staff, regulators"
    },
    "IoT System": {
        "system": "Smart IoT system managing connected sensors",
        "stakeholders": "Operators, engineers, users"
    }
}

DOMAIN_LIST = list(DOMAIN_CONTEXTS.keys())

# ==========================
# SYSTEM CONTEXT BUILDER
# ==========================
def create_system_context(domain: str, stakeholders: str, system_desc: str) -> str:
    return f"""System Domain: {domain}

System Description:
{system_desc}

Stakeholders:
{stakeholders}

The system must ensure:
- Security
- Performance
- Reliability
- Scalability
- Usability
- Compliance
- Maintainability"""

# ==========================
# TF-IDF CONTEXT BUILDER
# Used when classifying a story within a batch/dataset
# (finds similar neighboring stories to add as context)
# ==========================
def build_tfidf_context(df, index, tfidf_matrix, threshold=0.1, window_size=5):
    """
    Returns (previous_context, next_context) strings based on
    cosine similarity of neighboring requirements.
    """
    prev_reqs = []
    next_reqs = []
    current_vector = tfidf_matrix[index]

    for i in range(1, window_size + 1):
        if index - i >= 0:
            sim = cosine_similarity(current_vector, tfidf_matrix[index - i:index - i + 1])[0][0]
            if sim >= threshold:
                prev_reqs.append(df.iloc[index - i]["story"])
        if index + i < len(df):
            sim = cosine_similarity(current_vector, tfidf_matrix[index + i:index + i + 1])[0][0]
            if sim >= threshold:
                next_reqs.append(df.iloc[index + i]["story"])

    previous_context = "\n".join(prev_reqs[::-1]) if prev_reqs else "None"
    next_context     = "\n".join(next_reqs)       if next_reqs else "None"
    return previous_context, next_context

# ==========================
# CONTEXT PROMPT BUILDER
# ==========================
def create_context_prompt(
    previous_context: str,
    current_req: str,
    next_context: str,
    technique: str,
    system_context: str
) -> str:
    """
    Builds a domain-context-aware classification prompt.
    Includes previous/next requirements for surrounding context.
    """
    examples = ""
    if technique == "few_shot":
        examples = """
EXAMPLES
--------
Example 1:
Requirement: As a customer I want to place an order so that I can purchase products.
Is NFR: No
NFR Type: None

Example 2:
Requirement: The system shall encrypt all customer payment data.
Is NFR: Yes
NFR Type: Security

Example 3:
Requirement: The system shall respond to search queries within 2 seconds.
Is NFR: Yes
NFR Type: Performance
"""

    return f"""You are an expert software requirements engineer specializing in requirement classification.

SYSTEM CONTEXT
--------------
{system_context}

REQUIREMENT CONTEXT
-------------------
Previous Requirements:
{previous_context}

Current Requirement:
{current_req}

Next Requirements:
{next_context}
{examples}
TASK
----
Determine whether the CURRENT requirement is:

1. Functional Requirement (FR)
   - Describes system behavior, feature, or action.

2. Non-Functional Requirement (NFR)
   - Describes system qualities such as:
     performance, security, usability, reliability,
     scalability, availability, maintainability,
     privacy, safety, compliance.

INSTRUCTIONS
------------
• Analyze the requirement carefully.
• Consider surrounding requirements for context.
• Identify quality attributes if present.
• Only classify as NFR if it clearly represents a quality constraint.

OUTPUT FORMAT (STRICT)
----------------------
Respond ONLY in this exact format — no extra text:

Is NFR: Yes or No
NFR Type: <Performance | Security | Usability | Reliability | Scalability | Availability | Maintainability | Privacy | Safety | Compliance | None>
Reason: <one sentence>
"""

# ==========================
# RESPONSE EXTRACTION
# ==========================
def extract_is_nfr(response: str) -> str:
    """Returns 'Yes' or 'No'"""
    if not response:
        return "No"
    match = re.search(r"Is\s*NFR\s*:\s*(Yes|No)", response, re.IGNORECASE)
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"
    return "No"

def extract_nfr_type(response: str) -> str:
    """Extracts NFR type from model response"""
    match = re.search(
        r"NFR\s*Type\s*[:\-]\s*(Performance|Security|Usability|Reliability|Scalability|"
        r"Availability|Maintainability|Privacy|Safety|Compliance|None)",
        response, re.IGNORECASE
    )
    if match:
        return match.group(1).capitalize()
    return "None"

def extract_reason(response: str) -> str:
    """Extracts reason from model response"""
    match = re.search(r"Reason\s*:\s*(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "No reason provided"

def extract_rqi_score(response: str):
    """Extracts numeric RQI score (0–10) from LLM response"""
    text  = str(response)
    match = re.search(r'([0-9]*\.?[0-9]+)', text)
    if match:
        score = float(match.group(1))
        if 0 <= score <= 10:
            return score
    return None

# ==========================
# RQI — RULE-BASED (fast, no API cost)
# ==========================
def calculate_rqi_rule_based(requirement: str) -> int:
    """
    Scores a requirement on 15 quality criteria.
    Returns a score from 0–10.
    """
    if not requirement or not isinstance(requirement, str):
        return 0

    req   = requirement.lower().strip()
    score = 0

    # 1 Correctness
    if any(w in req for w in ["system", "application", "user"]) and \
       any(w in req for w in ["shall", "must"]):
        score += 1

    # 2 Clarity — no vague words
    if not any(w in req for w in ["fast", "quick", "efficient", "etc", "user-friendly"]):
        score += 1

    # 3 Completeness — has conditions
    if any(w in req for w in ["if", "when", "within", "under", "before", "after"]):
        score += 1

    # 4 Consistency
    if not ("must not" in req and "must" in req):
        score += 1

    # 5 Feasibility
    if not any(w in req for w in ["always", "never fail", "perfect", "100% secure"]):
        score += 1

    # 6 Verifiability — has measurable terms
    if any(w in req for w in ["seconds", "ms", "%", "response time", "latency"]):
        score += 1

    # 7 Traceability
    if any(w in req for w in ["shall", "must"]):
        score += 1

    # 8 Modifiability — concise
    if len(req.split()) < 40:
        score += 1

    # 9 Atomicity — single concern
    if req.count(" and ") <= 1:
        score += 1

    # 10 Structured Language
    if req.startswith(("the system", "the application", "the software")):
        score += 1

    # 11 Unambiguity
    if not any(w in req for w in ["maybe", "possibly", "should ideally"]):
        score += 1

    # 12 Testability
    if any(w in req for w in ["verify", "validate", "measure"]):
        score += 1

    # 13 Security Awareness
    if "encrypt" in req or "authentication" in req:
        score += 1

    # 14 User Story Format
    if "as a" in req and "i want" in req:
        score += 1

    # 15 Performance Constraint
    if any(w in req for w in ["within", "throughput", "latency"]):
        score += 1

    return round((score / 15) * 10)

# ==========================
# RQI — LLM-BASED
# Calls one or more models and averages their scores
# ==========================
def calculate_rqi_llm(requirement: str, model_fns: list) -> float:
    """
    model_fns: list of (model_fn, model_name) tuples
    Each model_fn(prompt, technique) should return a plain string.
    Returns average RQI score from all models, or None if all fail.
    """
    prompt = f"""Evaluate the quality of this software requirement.

Requirement:
{requirement}

Score it using Requirement Quality Index (RQI) from 0 to 10.
Return ONLY a number like:
RQI: 7.5
"""
    scores = []
    for model_fn, name in model_fns:
        try:
            response = model_fn(prompt, "zero_shot")
            score    = extract_rqi_score(response)
            if score is not None:
                scores.append(score)
        except Exception as e:
            print(f"RQI LLM error ({name}): {e}")

    return round(sum(scores) / len(scores), 1) if scores else None

# ==========================
# COMBINED RQI
# Blends rule-based + LLM scores
# ==========================
def calculate_combined_rqi(requirement: str, model_fns: list = None) -> dict:
    """
    Returns dict with rule_score, llm_score, final_score.
    If model_fns is None or empty, uses only rule-based.
    """
    rule_score = calculate_rqi_rule_based(requirement)
    llm_score  = None
    final      = float(rule_score)

    if model_fns:
        llm_score = calculate_rqi_llm(requirement, model_fns)
        if llm_score is not None:
            final = round((rule_score + llm_score) / 2, 1)

    # Generate quality label
    if final >= 8:
        label = "Excellent"
        color = "green"
    elif final >= 6:
        label = "Good"
        color = "blue"
    elif final >= 4:
        label = "Fair"
        color = "yellow"
    else:
        label = "Poor"
        color = "red"

    return {
        "rule_score": rule_score,
        "llm_score":  llm_score,
        "final_score": final,
        "label": label,
        "color": color
    }

# ==========================
# CRITERIA BREAKDOWN
# Returns which of the 15 criteria passed/failed
# ==========================
def get_rqi_criteria_breakdown(requirement: str) -> list:
    """
    Returns list of dicts: [{"name": ..., "passed": True/False, "description": ...}]
    """
    req = requirement.lower().strip()

    criteria = [
        {
            "name": "Correctness",
            "passed": any(w in req for w in ["system", "application", "user"]) and
                      any(w in req for w in ["shall", "must"]),
            "description": "References the system and uses 'shall'/'must'"
        },
        {
            "name": "Clarity",
            "passed": not any(w in req for w in ["fast", "quick", "efficient", "etc", "user-friendly"]),
            "description": "Avoids vague terms like 'fast', 'user-friendly'"
        },
        {
            "name": "Completeness",
            "passed": any(w in req for w in ["if", "when", "within", "under", "before", "after"]),
            "description": "Specifies conditions or constraints"
        },
        {
            "name": "Consistency",
            "passed": not ("must not" in req and "must" in req),
            "description": "No contradictory statements"
        },
        {
            "name": "Feasibility",
            "passed": not any(w in req for w in ["always", "never fail", "perfect", "100% secure"]),
            "description": "No unrealistic absolutes"
        },
        {
            "name": "Verifiability",
            "passed": any(w in req for w in ["seconds", "ms", "%", "response time", "latency"]),
            "description": "Contains measurable metrics"
        },
        {
            "name": "Traceability",
            "passed": any(w in req for w in ["shall", "must"]),
            "description": "Uses formal requirement language"
        },
        {
            "name": "Modifiability",
            "passed": len(req.split()) < 40,
            "description": "Concise — under 40 words"
        },
        {
            "name": "Atomicity",
            "passed": req.count(" and ") <= 1,
            "description": "Focuses on a single concern"
        },
        {
            "name": "Structured Language",
            "passed": req.startswith(("the system", "the application", "the software")),
            "description": "Starts with 'The system/application/software'"
        },
        {
            "name": "Unambiguity",
            "passed": not any(w in req for w in ["maybe", "possibly", "should ideally"]),
            "description": "No ambiguous qualifiers"
        },
        {
            "name": "Testability",
            "passed": any(w in req for w in ["verify", "validate", "measure"]),
            "description": "Can be tested or verified"
        },
        {
            "name": "Security Awareness",
            "passed": "encrypt" in req or "authentication" in req,
            "description": "Mentions security mechanisms"
        },
        {
            "name": "User Story Format",
            "passed": "as a" in req and "i want" in req,
            "description": "Follows 'As a... I want...' format"
        },
        {
            "name": "Performance Constraint",
            "passed": any(w in req for w in ["within", "throughput", "latency"]),
            "description": "Specifies performance bounds"
        },
    ]
    return criteria