import re

DOMAIN_CONTEXTS = {
    "Software Application": {
        "system": "Software system handling user accounts and transactions",
        "stakeholders": "End users, developers, administrators",
    },
    "Healthcare System": {
        "system": "Electronic health record system storing patient data",
        "stakeholders": "Doctors, nurses, hospital admins, patients",
    },
    "E-Commerce": {
        "system": "Online shopping platform managing products and orders",
        "stakeholders": "Customers, sellers, admins",
    },
    "Banking System": {
        "system": "Banking system handling financial transactions",
        "stakeholders": "Customers, bank staff, regulators",
    },
    "IoT System": {
        "system": "Smart IoT system managing connected sensors",
        "stakeholders": "Operators, engineers, users",
    },
}

DOMAIN_LIST = list(DOMAIN_CONTEXTS.keys())


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


def create_context_prompt(
    previous_context: str,
    current_req: str,
    next_context: str,
    technique: str,
    system_context: str,
) -> str:
    """
    Build a domain-aware classification prompt with neighboring requirement context.
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
- Analyze the requirement carefully.
- Consider surrounding requirements for context.
- Identify quality attributes if present.
- Only classify as NFR if it clearly represents a quality constraint.

OUTPUT FORMAT (STRICT)
----------------------
Respond ONLY in this exact format with no extra text:

Is NFR: Yes or No
NFR Type: <Performance | Security | Usability | Reliability | Scalability | Availability | Maintainability | Privacy | Safety | Compliance | None>
Reason: <one sentence>
"""


def extract_is_nfr(response: str) -> str:
    """Return 'Yes' or 'No' from an LLM response."""
    if not response:
        return "No"

    match = re.search(r"Is\s*NFR\s*:\s*(Yes|No)", response, re.IGNORECASE)
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"
    return "No"


def extract_nfr_type(response: str) -> str:
    """Extract the NFR type from a model response."""
    match = re.search(
        r"NFR\s*Type\s*[:\-]\s*(Performance|Security|Usability|Reliability|Scalability|"
        r"Availability|Maintainability|Privacy|Safety|Compliance|None)",
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).capitalize()
    return "None"


def extract_reason(response: str) -> str:
    """Extract the reason sentence from a model response."""
    match = re.search(r"Reason\s*:\s*(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "No reason provided"


def extract_confidence(response: str) -> int:
    """
    Extract a confidence score (0-100) from common LLM response formats.
    Falls back to 70 when confidence is not present.
    """
    if not response:
        return 70

    match = re.search(r"(?:4\.)?\s*Confidence\s*[:\-]\s*([0-9]{1,3})", response, re.IGNORECASE)
    if match:
        value = int(match.group(1))
        return min(100, max(0, value))

    match = re.search(r"\b([5-9][0-9]|100)\b\s*$", response, re.IGNORECASE | re.MULTILINE)
    if match:
        return int(match.group(1))

    return 70


def extract_rqi_score(response: str):
    """Extract a numeric RQI score (0-10) from model output."""
    text = str(response)
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    if match:
        score = float(match.group(1))
        if 0 <= score <= 10:
            return score
    return None


def calculate_rqi_rule_based(requirement: str) -> int:
    """
    Score a requirement on 15 quality criteria and return a 0-10 score.
    """
    if not requirement or not isinstance(requirement, str):
        return 0

    req = requirement.lower().strip()
    score = 0

    if any(word in req for word in ["system", "application", "user"]) and any(
        word in req for word in ["shall", "must"]
    ):
        score += 1

    if not any(word in req for word in ["fast", "quick", "efficient", "etc", "user-friendly"]):
        score += 1

    if any(word in req for word in ["if", "when", "within", "under", "before", "after"]):
        score += 1

    if not ("must not" in req and "must" in req):
        score += 1

    if not any(word in req for word in ["always", "never fail", "perfect", "100% secure"]):
        score += 1

    if any(word in req for word in ["seconds", "ms", "%", "response time", "latency"]):
        score += 1

    if any(word in req for word in ["shall", "must"]):
        score += 1

    if len(req.split()) < 40:
        score += 1

    if req.count(" and ") <= 1:
        score += 1

    if req.startswith(("the system", "the application", "the software")):
        score += 1

    if not any(word in req for word in ["maybe", "possibly", "should ideally"]):
        score += 1

    if any(word in req for word in ["verify", "validate", "measure"]):
        score += 1

    if "encrypt" in req or "authentication" in req:
        score += 1

    if "as a" in req and "i want" in req:
        score += 1

    if any(word in req for word in ["within", "throughput", "latency"]):
        score += 1

    return round((score / 15) * 10)


def calculate_rqi_llm(requirement: str, model_fns: list) -> float:
    """
    Call one or more model functions and return the average RQI score.
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
            score = extract_rqi_score(response)
            if score is not None:
                scores.append(score)
        except Exception as exc:
            print(f"RQI LLM error ({name}): {exc}")

    return round(sum(scores) / len(scores), 1) if scores else None


def calculate_combined_rqi(requirement: str, model_fns: list = None) -> dict:
    """
    Blend rule-based and LLM-based RQI into a single summary payload.
    """
    rule_score = calculate_rqi_rule_based(requirement)
    llm_score = None
    final_score = float(rule_score)

    if model_fns:
        llm_score = calculate_rqi_llm(requirement, model_fns)
        if llm_score is not None:
            final_score = round((rule_score + llm_score) / 2, 1)

    if final_score >= 8:
        label = "Excellent"
        color = "green"
    elif final_score >= 6:
        label = "Good"
        color = "blue"
    elif final_score >= 4:
        label = "Fair"
        color = "yellow"
    else:
        label = "Poor"
        color = "red"

    return {
        "rule_score": rule_score,
        "llm_score": llm_score,
        "final_score": final_score,
        "label": label,
        "color": color,
    }


def get_rqi_criteria_breakdown(requirement: str) -> list:
    """Return pass/fail details for the 15 RQI criteria."""
    req = requirement.lower().strip()

    return [
        {
            "name": "Correctness",
            "passed": any(word in req for word in ["system", "application", "user"])
            and any(word in req for word in ["shall", "must"]),
            "description": "References the system and uses 'shall'/'must'",
        },
        {
            "name": "Clarity",
            "passed": not any(word in req for word in ["fast", "quick", "efficient", "etc", "user-friendly"]),
            "description": "Avoids vague terms like 'fast' and 'user-friendly'",
        },
        {
            "name": "Completeness",
            "passed": any(word in req for word in ["if", "when", "within", "under", "before", "after"]),
            "description": "Specifies conditions or constraints",
        },
        {
            "name": "Consistency",
            "passed": not ("must not" in req and "must" in req),
            "description": "No contradictory statements",
        },
        {
            "name": "Feasibility",
            "passed": not any(word in req for word in ["always", "never fail", "perfect", "100% secure"]),
            "description": "No unrealistic absolutes",
        },
        {
            "name": "Verifiability",
            "passed": any(word in req for word in ["seconds", "ms", "%", "response time", "latency"]),
            "description": "Contains measurable metrics",
        },
        {
            "name": "Traceability",
            "passed": any(word in req for word in ["shall", "must"]),
            "description": "Uses formal requirement language",
        },
        {
            "name": "Modifiability",
            "passed": len(req.split()) < 40,
            "description": "Concise under 40 words",
        },
        {
            "name": "Atomicity",
            "passed": req.count(" and ") <= 1,
            "description": "Focuses on a single concern",
        },
        {
            "name": "Structured Language",
            "passed": req.startswith(("the system", "the application", "the software")),
            "description": "Starts with 'The system/application/software'",
        },
        {
            "name": "Unambiguity",
            "passed": not any(word in req for word in ["maybe", "possibly", "should ideally"]),
            "description": "No ambiguous qualifiers",
        },
        {
            "name": "Testability",
            "passed": any(word in req for word in ["verify", "validate", "measure"]),
            "description": "Can be tested or verified",
        },
        {
            "name": "Security Awareness",
            "passed": "encrypt" in req or "authentication" in req,
            "description": "Mentions security mechanisms",
        },
        {
            "name": "User Story Format",
            "passed": "as a" in req and "i want" in req,
            "description": "Follows 'As a... I want...' format",
        },
        {
            "name": "Performance Constraint",
            "passed": any(word in req for word in ["within", "throughput", "latency"]),
            "description": "Specifies performance bounds",
        },
    ]


__all__ = [
    "DOMAIN_CONTEXTS",
    "DOMAIN_LIST",
    "create_system_context",
    "create_context_prompt",
    "extract_is_nfr",
    "extract_nfr_type",
    "extract_reason",
    "extract_confidence",
    "extract_rqi_score",
    "calculate_rqi_rule_based",
    "calculate_rqi_llm",
    "calculate_combined_rqi",
    "get_rqi_criteria_breakdown",
]
