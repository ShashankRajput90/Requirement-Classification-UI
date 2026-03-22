import pandas as pd
import re
import json
import os
import hashlib
import seaborn as sns
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import time
from context_utils import (
    DOMAIN_CONTEXTS,
    create_system_context,
    build_tfidf_context as build_context,
    create_context_prompt,
    extract_is_nfr,
    extract_nfr_type,
    calculate_rqi_rule_based as calculate_rqi,
    calculate_rqi_llm as calculate_rqi_multi_llm,
    extract_rqi_score
)

# ==========================
# Rate limiting per model
# ==========================
rate_limit_delays = {
    "claude": 12,   # ~5 req/min
    "gemini": 15,   # ~5 req/min
    "cohere": 10,    # ~10 req/min
    "groq": 4,      # Groq is fast
    "mistral": 0    # local model
}

warnings.filterwarnings("ignore")
load_dotenv()

from code_integration import (
    classify_with_groq_deepseek,
    classify_with_groq,
    classify_with_gemini,
    classify_with_cohere,
    classify_with_claude,
    run_mistral_local
)

techniques = ["zero_shot", "few_shot"]
# ==========================
# RESULT DIRECTORIES
# ==========================
RESULT_DIR = "results"
PRED_DIR = os.path.join(RESULT_DIR, "predictions")
CM_DIR = os.path.join(RESULT_DIR, "confusion_matrices")
REPORT_DIR = os.path.join(RESULT_DIR, "classification_reports")
TYPEF1_DIR = os.path.join(RESULT_DIR, "per_type_f1")
METRICS_DIR = os.path.join(RESULT_DIR, "metrics")
RQI_DIR = os.path.join(RESULT_DIR, "rqi")

for folder in [RESULT_DIR, PRED_DIR, CM_DIR, REPORT_DIR, TYPEF1_DIR, METRICS_DIR, RQI_DIR]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------------
# DOMAIN CONTEXT TEMPLATES
# -----------------------------------
# DOMAIN_CONTEXTS = {

# "Software Application": {
#     "system":"Software system handling user accounts and transactions",
#     "stakeholders":"End users, developers, administrators"
# },

# "Healthcare System":{
#     "system":"Electronic health record system storing patient data",
#     "stakeholders":"Doctors, nurses, hospital admins, patients"
# },

# "E-Commerce":{
#     "system":"Online shopping platform managing products and orders",
#     "stakeholders":"Customers, sellers, admins"
# },

# "Banking System":{
#     "system":"Banking system handling financial transactions",
#     "stakeholders":"Customers, bank staff, regulators"
# },

# "IoT System":{
#     "system":"Smart IoT system managing connected sensors",
#     "stakeholders":"Operators, engineers, users"
# }
# }
# ==========================
# GLOBAL SYSTEM CONTEXT
# ==========================
# def create_system_context(domain, stakeholders, system_desc):
    
#     return f"""
# System Domain: {domain}

# System Description:
# {system_desc}

# Stakeholders:
# {stakeholders}

# The system must ensure:
# - Security
# - Performance
# - Reliability
# - Scalability
# - Usability
# - Compliance
# - Maintainability
# """
# ==========================
# CONTEXT BUILDER
# ==========================
# def build_context(df, index, tfidf_matrix, threshold=0.1, window_size=5):
#     prev_reqs = []
#     next_reqs = []

#     current_vector = tfidf_matrix[index]

#     for i in range(1, window_size + 1):
#         # Previous requirements
#         if index - i >= 0:
#             sim = cosine_similarity(
#                 current_vector,
#                 tfidf_matrix[index - i:index - i + 1]
#             )[0][0]
#             if sim >= threshold:
#                 prev_reqs.append(df.iloc[index - i]["story"])
#         # Next requirements
#         if index + i < len(df):
#             sim = cosine_similarity(
#                 current_vector,
#                 tfidf_matrix[index + i:index + i + 1]
#             )[0][0]

#             if sim >= threshold:
#                 next_reqs.append(df.iloc[index + i]["story"])

#     previous_context = "\n".join(prev_reqs[::-1]) if prev_reqs else "None"
#     next_context = "\n".join(next_reqs) if next_reqs else "None"

#     return previous_context, next_context

# ==========================
# PROMPT CREATION
# ==========================
# def create_context_prompt(previous_context, current_req, next_context, technique, system_context):
#     # Few-shot examples
#     if technique == "few_shot":
#         examples = """
# EXAMPLES
# --------
# Example 1:
# Requirement: As a customer I want to place an order so that I can purchase products.
# Is NFR: No
# NFR Type: None

# Example 2:
# Requirement: The system shall encrypt all customer payment data.
# Is NFR: Yes
# NFR Type: Security

# Example 3:
# Requirement: The system shall respond to search queries within 2 seconds.
# Is NFR: Yes
# NFR Type: Performance
# """
#     else:
#         examples = ""
        
#     prompt = f"""
# You are an expert software requirements engineer specializing in
# requirement classification.

# SYSTEM CONTEXT
# --------------
# {system_context}

# REQUIREMENT CONTEXT
# -------------------
# Previous Requirements:
# {previous_context}

# Current Requirement:
# {current_req}

# Next Requirements:
# {next_context}

# {examples}
# TASK
# ----
# Determine whether the CURRENT requirement is:

# 1. Functional Requirement (FR)
#    - Describes system behavior, feature, or action.

# 2. Non-Functional Requirement (NFR)
#    - Describes system qualities such as:
#      performance, security, usability, reliability,
#      scalability, availability, maintainability,
#      privacy, safety, compliance.

# INSTRUCTIONS
# ------------
# • Analyze the requirement carefully.
# • Consider surrounding requirements.
# • Identify quality attributes if present.
# • Only classify as NFR if it clearly represents a
#   quality constraint or system attribute.

# OUTPUT FORMAT (STRICT)
# ----------------------
# Respond ONLY in this format:

# Is NFR: Yes or No
# NFR Type: <Performance | Security | Usability | Reliability | Scalability | Availability | Maintainability | Privacy | Safety | Compliance | None>
# """
#     return prompt

# ==========================
# RQI PROMPT FOR LLM
# ==========================
def create_rqi_prompt(requirement):

    return f"""
You are a senior requirements engineering expert.

Evaluate the following requirement using these quality criteria:

1 Correctness
2 Clarity
3 Completeness
4 Consistency
5 Feasibility
6 Verifiability
7 Traceability
8 Modifiability
9 Atomicity
10 Structured language
11 Unambiguity
12 Testability
13 Security awareness
14 User story structure
15 Performance measurability

Score each criterion as 0 or 1.

Requirement:
{requirement}

Output format:

Correctness: X
Clarity: X
Completeness: X
Consistency: X
Feasibility: X
Verifiability: X
Traceability: X
Modifiability: X
Atomicity: X
Structure: X
Unambiguity: X
Testability: X
Security: X
UserStory: X
Performance: X

Final RQI score = (Sum of criteria / 15) * 10
Output the final score as:
Score the Requirement Quality Index (RQI) from 0 to 10.

Respond ONLY with a number.

Example:
7.5
"""
def classify_single_requirement(requirement, domain, model_fn, model_name, technique):
    
    context = DOMAIN_CONTEXTS.get(domain, DOMAIN_CONTEXTS["Software Application"])

    system_context = create_system_context(
        domain,
        context["stakeholders"],
        context["system"]
    )

    previous_context = "None"
    next_context = "None"

    prompt = create_context_prompt(
        previous_context,
        requirement,
        next_context,
        technique,
        system_context
    )

    response = model_fn(prompt, technique)

    is_nfr = extract_is_nfr(response)
    nfr_type = extract_nfr_type(response)

    print("\nRequirement:", requirement)
    print("Is NFR:", is_nfr)
    print("NFR Type:", nfr_type)

# ==========================
# EXTRACTION FUNCTIONS
# ==========================
# def extract_is_nfr(response):
#     match = re.search(r"Is\s*NFR\s*:\s*(Yes|No)", response, re.IGNORECASE)

#     if match:
#         result = match.group(1).lower()

#         if result == "yes":
#             return "Yes"
#         else:
#             return "No"

#     return "No"

# def extract_nfr_type(response):
    
#     match = re.search(
#         r"NFR\s*Type\s*[:\-]\s*(Performance|Security|Usability|Reliability|Scalability|Availability|Maintainability|Privacy|Safety|Compliance|None)",
#         response,
#         re.IGNORECASE
#     )

#     if match:
#         return match.group(1).capitalize()

#     return "None"

# def extract_rqi_score(response):
#     text = str(response)
#     match = re.search(r'([0-9]*\.?[0-9]+)', text)

#     if match:
#         score = float(match.group(1))

#         if 0 <= score <= 10:
#             return score

#     return None
# ==========================
# MULTI LLM RQI
# ==========================
def calculate_rqi_multi_llm(requirement, models):
    
    scores = []

    prompt = f"""
Evaluate the quality of this software requirement.

Requirement:
{requirement}

Score it using Requirement Quality Index (RQI) from 0 to 10.

Return ONLY in this format:
RQI: <number>
"""

    for model_fn, name in models:

        try:
            response = model_fn(prompt)

            print(f"{name} response:", response)

            score = extract_rqi_score(response)

            if score is not None:
                scores.append(score)

        except Exception as e:
            print(f"{name} error:", e)

    if len(scores) == 0:
        return None

    return sum(scores) / len(scores)

def safe_f1(p, r):
    return 0 if (p + r) == 0 else 2 * (p * r) / (p + r)

def get_cache_key(story, model_name, technique, system_context):
    return hashlib.md5(
        f"{story}-{model_name}-{technique}-{system_context}".encode()
    ).hexdigest()

# ==========================
# EVALUATION FUNCTION
# ==========================
def evaluate_model_with_context(model_fn, model_name, df, tfidf_matrix, cache_folder, batch_name, technique):
    os.makedirs(cache_folder, exist_ok=True)
    cache_file = os.path.join(cache_folder,f"context_cache_{model_name}_{technique}_{batch_name}.json")
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)

    y_true_bin = df["label"].astype(str).values

    responses = [None] * len(df)

    def get_response(index):
        story = df.loc[index, "story"]
        domain = df.iloc[index]["domain"]
        stakeholders = df.iloc[index]["stakeholders"]
        system_desc = df.iloc[index]["system"]

        system_context = create_system_context(
            domain,
            stakeholders,
            system_desc
        )
        previous_context, next_context = build_context(df,index,tfidf_matrix)
        prompt = create_context_prompt(previous_context, story, next_context, technique, system_context)
        key = get_cache_key(story, model_name, technique, system_context)

        if key in cache:
            return cache[key]

        try:
            response = model_fn(prompt, technique)
            # ==========================
            # Apply rate limiting
            # ==========================
            model_key = model_name.lower()

            for key in rate_limit_delays:
                if key in model_key:
                    time.sleep(rate_limit_delays[key])
                    break
                
            cache[key] = response
            return response
        except Exception as e:
            cache[key] = f"Error: {e}"
            return cache[key]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future_map = {
            executor.submit(get_response, idx): pos
            for pos, idx in enumerate(df.index)
        }

        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc=f"{model_name}-{technique}"
        ):
            pos = future_map[future]
            responses[pos] = future.result()

    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    y_pred_bin = []

    for r in responses:
        label = extract_is_nfr(r)
        if label not in ["Yes", "No"]:
            label = "No"
        y_pred_bin.append(label)
    
    y_true_bin = ["Yes" if str(x) == "Yes" else "No" for x in y_true_bin]
    y_pred_bin = ["Yes" if str(x) == "Yes" else "No" for x in y_pred_bin]
    
    y_pred_type = [extract_nfr_type(r) for r in responses]

    # ================= Binary Metrics =================
    binary_acc = accuracy_score(y_true_bin, y_pred_bin)
    binary_prec = precision_score(y_true_bin, y_pred_bin, pos_label="Yes", zero_division=0)
    binary_rec = recall_score(y_true_bin, y_pred_bin, pos_label="Yes", zero_division=0)
    binary_f1 = f1_score(y_true_bin, y_pred_bin, pos_label="Yes", zero_division=0)
    binary_f1_weighted = f1_score(y_true_bin,y_pred_bin,average="weighted",zero_division=0)

    # ================= Type Metrics (ONLY TRUE NFRs) =================
    mask = [label == "Yes" for label in y_true_bin]

    if sum(mask) == 0:
        type_prec = 0
        type_rec = 0
        type_f1 = 0
        y_true_bin_type = []
        y_pred_bin_type = []
    else:
        y_true_type = df.loc[mask, "nfr_type"].apply(
            lambda x: [t.strip() for t in re.split(r"[,/;|-]", str(x)) if t.strip()]
        ).tolist()

        y_pred_type_filtered = [y_pred_type[i] for i in range(len(mask)) if mask[i]]

        # Remove "None" labels safely
        y_true_type = [
            labels if labels != ["None"] else []
            for labels in y_true_type
        ]

        y_pred_type_filtered = [
            labels if labels != ["None"] else []
            for labels in y_pred_type_filtered
        ]

        mlb = MultiLabelBinarizer()
        mlb.fit(y_true_type + y_pred_type_filtered)

        y_true_bin_type = mlb.transform(y_true_type)
        y_pred_bin_type = mlb.transform(y_pred_type_filtered)

        type_prec = precision_score(
            y_true_bin_type,
            y_pred_bin_type,
            average="macro",
            zero_division=0
        )

        type_rec = recall_score(
            y_true_bin_type,
            y_pred_bin_type,
            average="macro",
            zero_division=0
        )

        type_f1 = safe_f1(type_prec, type_rec)
    # ==========================================================
    # SAVE CONFUSION MATRIX (BINARY)
    # ==========================================================
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=["No", "Yes"])

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FR","NFR"], yticklabels=["FR","NFR"])
    plt.title(f"Binary Confusion Matrix\n{model_name}-{technique}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_filename = f"context_confusion_matrix_{model_name}_{technique}_{batch_name}"
    cm_filename = cm_filename.replace(".csv", "") + ".png"

    plt.savefig(os.path.join(CM_DIR, cm_filename))
    plt.close()

    # ==========================================================
    # SAVE CLASSIFICATION REPORT (TYPE LEVEL)
    # ==========================================================
    if len(y_true_bin_type) > 0:
        report_dict = classification_report(y_true_bin_type, y_pred_bin_type, output_dict=True, zero_division=0)

        report = classification_report(y_true_bin_type, y_pred_bin_type, zero_division=0)

        # Save per-class F1
        type_f1_per_class = {
            label: round(report_dict[label]["f1-score"], 3)
            for label in report_dict
            if label not in ["micro avg", "macro avg", "weighted avg"]
        }

        per_type_filename = f"context_per_type_f1_{model_name}_{technique}_{batch_name}"
        per_type_filename = per_type_filename.replace(".csv", "") + ".json"

        with open(os.path.join(TYPEF1_DIR, per_type_filename), "w") as f:
            json.dump(type_f1_per_class, f, indent=2)
    else:
        report = "No NFR samples in this batch."
        
    report_filename = f"context_classification_report_{model_name}_{technique}_{batch_name}"
    report_filename = report_filename.replace(".csv", "") + ".txt"
    
    with open(os.path.join(REPORT_DIR, report_filename), "w") as f:
        f.write(report)

    # ==========================================================
    # RETURN FULL METRICS
    # ==========================================================
    full_binary_metrics = {
        "Batch": batch_name,
        "Model": model_name,
        "Technique": technique,
        "Accuracy": round(binary_acc, 3),
        "Precision": round(binary_prec, 3),
        "Recall": round(binary_rec, 3),
        "Binary_F1": round(binary_f1, 3),
        "Binary_F1_Weighted": round(binary_f1_weighted, 3)
    }

    full_type_metrics = {
        "Batch": batch_name,
        "Model": model_name,
        "Technique": technique,
        "Type_F1": round(type_f1, 3)
    }
    
    # ================= SAVE RAW PREDICTIONS =================
    prediction_df = pd.DataFrame({
    "Story": df["story"],
    "True_Label": y_true_bin,
    "Pred_Label": y_pred_bin,
    "True_Type": df["nfr_type"],
    "Pred_Type": y_pred_type,
    "LLM_Response": responses
    })
    prediction_filename = f"context_predictions_{model_name}_{technique}_{batch_name}"
    prediction_filename = prediction_filename.replace(".csv", "") + ".csv"

    prediction_df.to_csv(os.path.join(PRED_DIR, prediction_filename), index=False)

    return full_binary_metrics, full_type_metrics

# ==========================
# EXTENDED RQI (15 Criteria)
# ==========================
# def calculate_rqi(requirement: str) -> int:

#     if not requirement or not isinstance(requirement, str):
#         return 0

#     req = requirement.lower().strip()
#     score = 0
#     total_criteria = 15

#     # 1 Correctness
#     if any(w in req for w in ["system", "application", "user"]) and any(w in req for w in ["shall", "must"]):
#         score += 1

#     # 2 Clarity
#     vague_words = ["fast", "quick", "efficient", "etc", "user-friendly"]
#     if not any(w in req for w in vague_words):
#         score += 1

#     # 3 Completeness
#     if any(w in req for w in ["if", "when", "within", "under", "before", "after"]):
#         score += 1

#     # 4 Consistency
#     if not ("must not" in req and "must" in req):
#         score += 1

#     # 5 Feasibility
#     unrealistic = ["always", "never fail", "perfect", "100% secure"]
#     if not any(w in req for w in unrealistic):
#         score += 1

#     # 6 Verifiability
#     measurable = ["seconds", "ms", "%", "response time", "latency"]
#     if any(w in req for w in measurable):
#         score += 1

#     # 7 Traceability
#     if any(w in req for w in ["shall", "must"]):
#         score += 1

#     # 8 Modifiability
#     if len(req.split()) < 40:
#         score += 1

#     # 9 Atomicity
#     if req.count(" and ") <= 1:
#         score += 1

#     # 10 Structured Language
#     if req.startswith(("the system", "the application", "the software")):
#         score += 1

#     # 11 Ambiguity Avoidance
#     ambiguous = ["maybe", "possibly", "should ideally"]
#     if not any(w in req for w in ambiguous):
#         score += 1

#     # 12 Testability
#     if any(w in req for w in ["verify", "validate", "measure"]):
#         score += 1

#     # 13 Security Awareness
#     if "encrypt" in req or "authentication" in req:
#         score += 1

#     # 14 User Story Format
#     if "as a" in req and "i want" in req:
#         score += 1

#     # 15 Performance Constraint
#     if any(w in req for w in ["within", "throughput", "latency"]):
#         score += 1

#     return round((score / total_criteria) * 10)
# ==========================
# RQI STATISTICS
# ==========================
def compute_rqi_statistics(df, batch_name):

    stats = {
        "Batch": batch_name,
        "Total_Requirements": len(df),
        "Average_RQI": round(df["RQI_Score"].mean(), 2),
        "Median_RQI": round(df["RQI_Score"].median(), 2),
        "Max_RQI": df["RQI_Score"].max(),
        "Min_RQI": df["RQI_Score"].min(),
        "Std_RQI": round(df["RQI_Score"].std(), 2)
    }

    stats_df = pd.DataFrame([stats])

    stats_df.to_csv(
        os.path.join(RQI_DIR, f"RQI_statistics_{batch_name}.csv"),
        index=False
    )

    print("RQI statistics saved")
    
def plot_rqi_distribution(df, batch_name):
    
    plt.figure()

    sns.histplot(df["RQI_Score"], bins=10)

    plt.title("RQI Score Distribution")
    plt.xlabel("RQI Score")
    plt.ylabel("Frequency")

    plt.savefig(
        os.path.join(RQI_DIR, f"RQI_distribution_{batch_name}.png")
    )

    plt.close()
# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    print("\n==============================")
    print("NFR Context-Based Classifier")
    print("==============================")
    print("1. Classify Single Requirement")
    print("2. Run Dataset Evaluation")
    print("==============================")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":

        requirement = input("\nEnter Requirement:\n")

        print("\nAvailable Domains:")
        for d in DOMAIN_CONTEXTS.keys():
            print("-", d)

        domain = input("\nEnter Domain: ").strip()

        print("\nSelect Model:")
        print("1. OpenAI GPT-OSS 120B")
        print("2. Groq LLaMA3.1 8B")
        print("3. Gemini 2.5 Pro")
        print("4. Cohere Command-R+")
        print("5. Claude 3 Haiku")
        print("6. Mistral Local")

        model_choice = input("\nEnter model number: ").strip()

        model_map = {
            "1": (classify_with_groq_deepseek, "OpenAI GPT-OSS 120B"),
            "2": (classify_with_groq, "Groq LLaMA3.1 8B"),
            "3": (classify_with_gemini, "Gemini 2.5 Pro"),
            "4": (classify_with_cohere, "Cohere Command-R+"),
            "5": (classify_with_claude, "Claude 3 Haiku"),
            "6": (run_mistral_local, "Mistral Local")
        }

        model_fn, model_name = model_map.get(
            model_choice,
            (classify_with_groq, "Groq LLaMA3.1 8B")
        )

        technique = input("\nEnter Prompt Technique (zero_shot / few_shot): ").strip()

        classify_single_requirement(
            requirement,
            domain,
            model_fn,
            model_name,
            technique
        )

        print("\n✅ Single Requirement Classification Complete!")

    elif choice == "2":

        print("\nRunning Dataset Evaluation...\n")

        all_binary_results = []
        all_type_results = []
        all_rqi_results = []

        cache_folder = "context_cache"

        models = [
            (classify_with_groq_deepseek, "OpenAI GPT-OSS 120B"),
            (classify_with_groq, "Groq LLaMA3.1 8B"),
            (classify_with_gemini, "Gemini 2.5 Pro"),
            (classify_with_cohere, "Cohere Command-R+"),
            (classify_with_claude, "Claude 3 Haiku"),
            (run_mistral_local, "Mistral Local"),
        ]

        rqi_models = [
        (classify_with_claude, "Claude 3 Haiku"),
        (classify_with_cohere, "Cohere Command-R+"),
        ]
        dataset_file = "dataset/strong_100_user_stories_dataset.csv"

        df = pd.read_csv(dataset_file)

        batch_file = os.path.basename(dataset_file)
        batch_file = batch_file.replace(".csv", "")

        print("Dataset Loaded Successfully")
        print(df.head())

        # ==========================
        # Compute RQI
        # ==========================
        print("\nCalculating Rule-Based RQI...")
        df["RQI_Rule"] = df["story"].apply(calculate_rqi)

        print("Calculating Multi-LLM RQI...")
        df["RQI_LLM"] = df["story"].apply(
        lambda x: calculate_rqi_multi_llm(x, rqi_models)
        )

        df["RQI_Score"] = (df["RQI_Rule"] + df["RQI_LLM"]) / 2
        rqi_file = os.path.join(RQI_DIR, f"RQI_detailed_{batch_file}.csv")

        df.to_csv(rqi_file, index=False)

        print("RQI detailed results saved")
        # ==========================
        # TF-IDF Context Setup
        # ==========================
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(df["story"].fillna(""))

        # ==========================
        # Normalize Labels
        # ==========================
        
        df["label"] = df["label"].astype(str).str.strip()

        df["label"] = df["label"].apply( lambda x: "Yes" if x.lower() in ["yes", "nfr"] else "No")

        print("\nLabel Distribution:")
        print(df["label"].value_counts())

        df["nfr_type"] = df["nfr_type"].fillna("None")

        # ==========================
        # MODEL EVALUATION
        # ==========================
        for model_fn, model_name in models:

            for technique in techniques:

                print(f"\nRunning Model: {model_name} | Technique: {technique}")

                binary_res, type_res = evaluate_model_with_context(
                    model_fn,
                    model_name,
                    df,
                    tfidf_matrix,
                    cache_folder,
                    batch_file,
                    technique
                )

                all_binary_results.append(binary_res)
                all_type_results.append(type_res)

        # ==========================
        # SAVE RESULTS
        # ==========================
        pd.DataFrame(all_binary_results).to_csv(
            os.path.join(METRICS_DIR, "FINAL_context_binary_metrics.csv"),
            index=False
        )

        pd.DataFrame(all_type_results).to_csv(
            os.path.join(METRICS_DIR, "FINAL_context_type_metrics.csv"),
            index=False
        )

        df.to_csv(os.path.join(RQI_DIR, f"RQI_detailed_{batch_file}.csv"), index=False)
        compute_rqi_statistics(df, batch_file)
        plot_rqi_distribution(df, batch_file)
        
        print("\n✅ Context-Based Requirement Analysis Complete!")

    else:
        print("\n❌ Invalid Choice. Please run again.")