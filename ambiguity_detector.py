import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AmbiguityWarning:
    code: str
    severity: str          # "error" | "warning" | "info"
    category: str
    title: str
    message: str
    suggestion: str
    matched_text: Optional[str] = None


@dataclass
class AmbiguityReport:
    requirement: str
    warnings: List[AmbiguityWarning] = field(default_factory=list)
    ambiguity_score: int = 0
    quality_label: str = "Good"
    is_ambiguous: bool = False


# ── Rule tables ───────────────────────────────────────────────
VAGUE_TERMS = [
    "fast","quickly","slow","efficient","user-friendly","intuitive","easy",
    "simple","nice","good","better","best","appropriate","sufficient",
    "adequate","reasonable","flexible","robust","modern","seamless","smooth",
    "optimal","optimized","clean","lightweight","powerful","advanced","smart",
    "intelligent","enhanced","improved","etc","and so on","and more",
    "and others","etcetera",
]
ABSOLUTE_TERMS = [
    "always","never","all users","every user","zero downtime","perfect",
    "perfectly","flawless","100 percent","100%","completely secure","totally",
    "fully automated","at all times","without any exception","instantly",
    "immediately",
]
WEAK_MODALS   = [r"\bshould\b",r"\bcould\b",r"\bmay\b",r"\bmight\b",r"\bwould\b"]
STRONG_MODALS = [r"\bshall\b",r"\bmust\b",r"\bwill\b"]

NFR_QUALITY_KEYWORDS = [
    "performance","fast","speed","latency","throughput","responsive",
    "available","availability","reliable","reliability","secure","security",
    "scalable","scalability",
]
METRIC_PATTERNS = [
    r"\d+\s*(ms|milliseconds?|seconds?|s\b|minutes?|hours?)",
    r"\d+\s*%",
    r"\d+\s*(req|requests?|transactions?|users?|concurrent)",
    r"\d+\s*(mb|gb|kb|tb)",
    r"\b(within|less than|at least|no more than|up to)\s+\d+",
    r"[≤≥<>]\s*\d+",
]
PASSIVE_PATTERNS = [
    r"\bshould be able to\b",r"\bwill be\b",r"\bmight be\b",
    r"\bneeds to be\b",r"\bit is (?:required|expected|needed)\b",r"\bhas to be\b",
]
DOUBLE_NEGATION_PATTERNS = [
    r"\bnot\b.{0,30}\bnot\b",r"\bno\b.{0,20}\bwithout\b",r"\bnever\b.{0,20}\bunless\b",
]
IMPL_LEAK_PATTERNS = [
    (r"\busing (?:mysql|postgresql|mongodb|redis|java|python|react|angular|vue"
     r"|aws|azure|gcp|docker|kubernetes|django|flask|spring)\b"),
    r"\bimplemented (?:in|with|using)\b",
    r"\bvia (?:rest api|soap|grpc|graphql|http|tcp|udp)\b",
    r"\bstored in\b",r"\bwritten in\b",
]
ACTOR_PATTERNS = [
    r"\bas (?:a|an) [\w\s]+,?\s*i (?:want|need|would like|can)\b",
    r"\bthe (?:system|application|software|platform|service)\b",
    r"\bthe (?:user|admin|customer|operator|manager|developer|client)\b",
]
CONDITION_PATTERNS = [
    r"\bwhen\b",r"\bif\b",r"\bafter\b",r"\bbefore\b",
    r"\bupon\b",r"\bon (?:success|failure|error|login|submit)\b",
]
PRONOUN_PATTERNS = [
    r"\bit\s+(?:shall|must|should|will|can)\b",
    r"\bthey\s+(?:shall|must|should|will|can)\b",
    r"\bthis\s+(?:shall|must|should|will|can)\b",
]
TBD_PATTERNS = [
    r"\btbd\b",r"\bto be defined\b",r"\bto be determined\b",
    r"\bplaceholder\b",r"\bxxx\b",
]
MIN_WORDS = 5
MAX_WORDS = 80


def detect_ambiguity(requirement: str) -> AmbiguityReport:
    if not requirement or not isinstance(requirement, str):
        return AmbiguityReport(
            requirement=requirement or "",
            warnings=[AmbiguityWarning(
                code="EMPTY", severity="error", category="Completeness",
                title="Empty Requirement",
                message="The requirement text is empty.",
                suggestion="Provide a complete, non-empty requirement statement.",
            )],
            ambiguity_score=100, quality_label="Critical", is_ambiguous=True,
        )

    req = requirement.strip()
    req_lower = req.lower()
    words = req_lower.split()
    report = AmbiguityReport(requirement=req)

    # 1. Vague terms
    for term in VAGUE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", req_lower):
            report.warnings.append(AmbiguityWarning(
                code="VAGUE_TERM", severity="warning", category="Clarity",
                title="Vague / Subjective Term",
                message=f'"{term}" is subjective and cannot be measured or tested.',
                suggestion=f'Replace "{term}" with a concrete criterion, e.g., "within 200 ms" or "≥ 99.9% uptime".',
                matched_text=term,
            ))

    # 2. Absolutes
    for term in ABSOLUTE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", req_lower):
            report.warnings.append(AmbiguityWarning(
                code="ABSOLUTE_TERM", severity="error", category="Feasibility",
                title="Unrealistic Absolute",
                message=f'"{term}" is an absolute that is rarely achievable.',
                suggestion='Replace with a realistic bound, e.g., "99.9% of the time".',
                matched_text=term,
            ))

    # 3. Weak modal
    has_strong = any(re.search(p, req_lower) for p in STRONG_MODALS)
    weak_hits = [p for p in WEAK_MODALS if re.search(p, req_lower)]
    if weak_hits and not has_strong:
        m = re.search(weak_hits[0], req_lower)
        report.warnings.append(AmbiguityWarning(
            code="WEAK_MODAL", severity="warning", category="Precision",
            title="Weak Modal Verb",
            message='"should/could/may/might" makes the requirement optional, not mandatory.',
            suggestion='Use "shall" or "must" for mandatory requirements.',
            matched_text=m.group(0) if m else None,
        ))

    # 4. NFR without metric
    has_nfr_kw = any(re.search(rf"\b{re.escape(kw)}\b", req_lower) for kw in NFR_QUALITY_KEYWORDS)
    has_metric = any(re.search(p, req_lower) for p in METRIC_PATTERNS)
    if has_nfr_kw and not has_metric:
        report.warnings.append(AmbiguityWarning(
            code="MISSING_METRIC", severity="error", category="Verifiability",
            title="NFR Without Measurable Metric",
            message="A quality attribute is mentioned but no numeric threshold is given.",
            suggestion='Add a metric, e.g., "…within 2 s", "≥ 99.5% uptime", "< 500 ms".',
        ))

    # 5. Compound
    if req_lower.count(" and ") >= 2:
        report.warnings.append(AmbiguityWarning(
            code="COMPOUND_REQ", severity="warning", category="Atomicity",
            title="Compound / Multi-Concern Requirement",
            message="Multiple 'and' occurrences suggest this covers more than one concern.",
            suggestion="Split into separate atomic requirements.",
        ))

    # 6. Missing actor
    if not any(re.search(p, req_lower) for p in ACTOR_PATTERNS):
        report.warnings.append(AmbiguityWarning(
            code="MISSING_ACTOR", severity="info", category="Completeness",
            title="Missing Actor / Subject",
            message="No clear subject (user, system, admin) is identified.",
            suggestion='Start with "The system shall…", "As a [role], I want…".',
        ))

    # 7. Double negation
    for pat in DOUBLE_NEGATION_PATTERNS:
        m = re.search(pat, req_lower)
        if m:
            report.warnings.append(AmbiguityWarning(
                code="DOUBLE_NEGATION", severity="error", category="Clarity",
                title="Double Negation",
                message="Double negations are confusing and easy to misinterpret.",
                suggestion="Rewrite as a clear, positive statement.",
                matched_text=m.group(0),
            ))
            break

    # 8. Passive voice
    for pat in PASSIVE_PATTERNS:
        m = re.search(pat, req_lower)
        if m:
            report.warnings.append(AmbiguityWarning(
                code="PASSIVE_VOICE", severity="info", category="Clarity",
                title="Passive / Weak Voice",
                message=f'"{m.group(0)}" uses passive voice that obscures responsibility.',
                suggestion='Use active voice: "The system shall…".',
                matched_text=m.group(0),
            ))
            break

    # 9. Impl leak
    for pat in IMPL_LEAK_PATTERNS:
        m = re.search(pat, req_lower)
        if m:
            report.warnings.append(AmbiguityWarning(
                code="IMPL_LEAK", severity="info", category="Abstraction",
                title="Implementation Detail Leaked",
                message=f'Specifies a technology: "{m.group(0)}".',
                suggestion="State WHAT, not HOW. Move technology choices to design docs.",
                matched_text=m.group(0),
            ))
            break

    # 10. Ambiguous pronoun
    for pat in PRONOUN_PATTERNS:
        m = re.search(pat, req_lower)
        if m:
            report.warnings.append(AmbiguityWarning(
                code="AMBIGUOUS_PRONOUN", severity="warning", category="Clarity",
                title="Ambiguous Pronoun",
                message=f'"{m.group(0)}" is unclear — replace the pronoun with the explicit noun.',
                suggestion="Replace pronouns with the explicit noun they refer to.",
                matched_text=m.group(0),
            ))
            break

    # 11. Length
    wc = len(words)
    if wc < MIN_WORDS:
        report.warnings.append(AmbiguityWarning(
            code="TOO_SHORT", severity="error", category="Completeness",
            title="Requirement Too Short",
            message=f"Only {wc} word(s) — likely incomplete.",
            suggestion="Provide a full requirement: subject + action + constraint.",
        ))
    elif wc > MAX_WORDS:
        report.warnings.append(AmbiguityWarning(
            code="TOO_LONG", severity="warning", category="Atomicity",
            title="Requirement Too Long",
            message=f"{wc} words — hard to test and trace.",
            suggestion="Break into smaller requirements (target < 40 words each).",
        ))

    # 12. Missing condition
    is_formal = bool(re.search(r"\b(?:shall|must)\b", req_lower))
    if is_formal and wc > 12 and not any(re.search(p, req_lower) for p in CONDITION_PATTERNS):
        report.warnings.append(AmbiguityWarning(
            code="MISSING_CONDITION", severity="info", category="Completeness",
            title="No Trigger / Condition Specified",
            message="No conditional context (when/if/after/upon) is present.",
            suggestion='Consider: "When [event], the system shall…".',
        ))

    # 13. TBD
    for pat in TBD_PATTERNS:
        m = re.search(pat, req_lower)
        if m:
            report.warnings.append(AmbiguityWarning(
                code="TBD_PLACEHOLDER", severity="error", category="Completeness",
                title="Undefined / TBD Value",
                message=f'Contains a placeholder: "{m.group(0)}".',
                suggestion="Replace with a concrete, agreed-upon value.",
                matched_text=m.group(0),
            ))
            break

    # De-duplicate
    seen, unique = set(), []
    for w in report.warnings:
        if w.code not in seen:
            seen.add(w.code)
            unique.append(w)
    report.warnings = unique

    # Score
    weights = {"error": 20, "warning": 10, "info": 4}
    raw = sum(weights.get(w.severity, 5) for w in report.warnings)
    report.ambiguity_score = min(100, raw)

    if report.ambiguity_score == 0:    report.quality_label, report.is_ambiguous = "Good",     False
    elif report.ambiguity_score <= 15: report.quality_label, report.is_ambiguous = "Good",     False
    elif report.ambiguity_score <= 35: report.quality_label, report.is_ambiguous = "Fair",     True
    elif report.ambiguity_score <= 60: report.quality_label, report.is_ambiguous = "Poor",     True
    else:                              report.quality_label, report.is_ambiguous = "Critical", True

    return report


def report_to_dict(report: AmbiguityReport) -> dict:
    return {
        "requirement":     report.requirement,
        "ambiguity_score": report.ambiguity_score,
        "quality_label":   report.quality_label,
        "is_ambiguous":    report.is_ambiguous,
        "total_warnings":  len(report.warnings),
        "error_count":     sum(1 for w in report.warnings if w.severity == "error"),
        "warning_count":   sum(1 for w in report.warnings if w.severity == "warning"),
        "info_count":      sum(1 for w in report.warnings if w.severity == "info"),
        "warnings": [
            {"code": w.code, "severity": w.severity, "category": w.category,
             "title": w.title, "message": w.message, "suggestion": w.suggestion,
             "matched_text": w.matched_text}
            for w in report.warnings
        ],
    }