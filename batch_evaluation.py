import pandas as pd
import re
import json
import os
import hashlib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# ==========================
# Import model functions
# ==========================
from code_integration import (
    classify_with_groq_deepseek,
    classify_with_groq,
    classify_with_gemini,
    classify_with_cohere,
    classify_with_claude,
    classify_with_mistral_local
)

techniques = ["zero_shot", "few_shot"]

# ==========================
# Helper Functions
# ==========================
def extract_is_nfr(response: str) -> str:
    """Extracts whether it's an NFR or not."""
    if not response:
        return "No"
    response = response.strip()
    # match = re.search(r"\b(?:is\s*)?(?:it\s*)?nfr\s*[:=\-]?\s*(yes|no)\b", response, re.IGNORECASE)
    match = re.search(r"1\.?\s*Is NFR:\s*(Yes|No)", response, re.IGNORECASE)
    if match:
        return "Yes" if match.group(1).strip().lower() == "yes" else "No"
    if re.search(r"\bnon[- ]?functional\b", response, re.IGNORECASE):
        return "Yes"
    if re.search(r"\bfunctional\b", response, re.IGNORECASE):
        return "No"
    return "No"

def extract_nfr_type_multilabel(response: str):
    """Extracts NFR type(s) as a list."""
    if not response:
        return ["Unknown"]
    match = re.search(r"2\.?\s*NFR Type:\s*(.*)", response, re.IGNORECASE)
    types_str = match.group(1) if match else response
    tokens = re.split(r"[,/;|-]", types_str)
    tokens = [t.strip() for t in tokens if t.strip()]

    keyword_map = {
        "Performance": ["performance", "speed", "latency", "throughput"],
        "Security": ["security", "secure", "authentication", "encryption"],
        "Usability": ["usability", "user-friendly", "ease of use", "ux"],
        "Reliability": ["reliability", "robustness", "dependability"],
        "Maintainability": ["maintainability", "modularity", "extensibility"],
        "Scalability": ["scalability", "scalable"],
        "Availability": ["availability", "uptime"],
        "Accessibility": ["accessibility", "inclusive design"],
        "Interoperability": ["interoperability", "compatibility"],
        "Portability": ["portability", "cross-platform"],
        "Quality": ["quality"],
        "Testability": ["testability"],
        "Compliance": ["compliance", "regulation"],
        "Accuracy": ["accuracy", "precision"],
        "Safety": ["safety"],
        "Efficiency": ["efficiency"],
        "Adaptability": ["adaptability", "flexibility"],
        "Robustness": ["robustness"],
        "Explainability": ["explainability", "interpretability"],
        "Transparency": ["transparency"],
        "Privacy": ["privacy"],
        "Integrity": ["integrity"],
    }

    detected = []
    for token in tokens:
        for label, keywords in keyword_map.items():
            if any(re.search(rf"\b{kw}\b", token, re.IGNORECASE) for kw in keywords):
                detected.append(label)
                break
    return detected if detected else ["Unknown"]

def get_cache_key(story: str, model_name: str, technique: str) -> str:
    """Unique cache key for each (story, model, technique)"""
    return hashlib.md5(f"{story}-{model_name}-{technique}".encode()).hexdigest()

def safe_f1(p, r):
    """Avoids divide-by-zero"""
    return 0.0 if (p + r) == 0 else 2 * (p * r) / (p + r)

# ==========================
# Evaluation Function
# ==========================
def evaluate_model(model_fn, model_name, df, cache_dir, batch_name, technique="zero_shot", tqdm_desc=None):
    batch_cache_dir = os.path.join(cache_dir, batch_name)
    os.makedirs(batch_cache_dir, exist_ok=True)
    cache_file = os.path.join(batch_cache_dir, f"cache_{model_name.replace(' ', '_')}_{technique}.json")

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)

    y_true_bin = df["label"].astype(str).str.strip().str.capitalize().replace({
        "Functional": "No", "Non-functional": "Yes", "Fr": "No", "Nfr": "Yes", "0": "No", "1": "Yes"
    }).values

    y_true_type = df["nfr_type"].fillna("None").astype(str)
    y_true_type = y_true_type.apply(lambda x: [t.strip() for t in re.split(r"[,/;|-]", x) if t.strip()])

    y_pred_bin, y_pred_type = [], []

    def get_response(story):
        key = get_cache_key(story, model_name, technique)
        if key in cache and not str(cache[key]).startswith("❌"):
            return cache[key]
        try:
            response = model_fn(story, technique)
            cache[key] = response
            return response
        except Exception as e:
            cache[key] = f"❌ {e}"
            return cache[key]

    # Rate limiting
    rate_limit_delays = {
        "claude": 12,
        "gemini": 12,
        "cohere": 6,
    }
    model_key = model_name.lower()
    delay = next((v for k, v in rate_limit_delays.items() if k in model_key), 0)

    # Run predictions
    if delay > 0:
        for story in tqdm(df["story"], desc=tqdm_desc, unit="req"):
            resp = get_response(story)
            y_pred_bin.append(extract_is_nfr(resp))
            y_pred_type.append(extract_nfr_type_multilabel(resp))
            time.sleep(delay)
    else:
        futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for story in df["story"]:
                futures.append(executor.submit(get_response, story))
            for future in tqdm(as_completed(futures), total=len(futures), desc=tqdm_desc, unit="req"):
                resp = future.result()
                y_pred_bin.append(extract_is_nfr(resp))
                y_pred_type.append(extract_nfr_type_multilabel(resp))

    # Save cache
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    # Binary metrics
    filtered_true, filtered_pred = zip(*[(yt, yp) for yt, yp in zip(y_true_bin, y_pred_bin)
                                         if yp not in ["Error", "❌", ""]])
    filtered_true = ["Yes" if "yes" in v.lower() else "No" for v in filtered_true]
    filtered_pred = ["Yes" if "yes" in v.lower() else "No" for v in filtered_pred]

    binary_acc = accuracy_score(filtered_true, filtered_pred)
    binary_prec = precision_score(filtered_true, filtered_pred, pos_label="Yes", zero_division=0)
    binary_rec = recall_score(filtered_true, filtered_pred, pos_label="Yes", zero_division=0)
    binary_f1 = safe_f1(binary_prec, binary_rec)

    # Type metrics
    y_true_type_filtered = [[t for t in lst if t != "Unknown"] for lst in y_true_type]
    y_pred_type_filtered = [[t for t in lst if t != "Unknown"] for lst in y_pred_type]

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_type_filtered + y_pred_type_filtered)
    y_true_bin_type = mlb.transform(y_true_type_filtered)
    y_pred_bin_type = mlb.transform(y_pred_type_filtered)

    type_acc = accuracy_score(y_true_bin_type, y_pred_bin_type)
    type_prec = precision_score(y_true_bin_type, y_pred_bin_type, average="macro", zero_division=0)
    type_rec = recall_score(y_true_bin_type, y_pred_bin_type, average="macro", zero_division=0)
    type_f1 = safe_f1(type_prec, type_rec)

    print(f"✅ {model_name} ({technique}) → Binary F1: {binary_f1:.2f} | Type F1: {type_f1:.2f}")

    binary_results = {
        "Batch": batch_name, "Model": model_name, "Technique": technique,
        "Accuracy": round(binary_acc, 3), "Precision": round(binary_prec, 3),
        "Recall": round(binary_rec, 3), "F1": round(binary_f1, 3)
    }

    type_results = {
        "Batch": batch_name, "Model": model_name, "Technique": technique,
        "Accuracy": round(type_acc, 3), "Precision": round(type_prec, 3),
        "Recall": round(type_rec, 3), "F1": round(type_f1, 3)
    }

    return binary_results, type_results

# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    batch_folder = "batches"
    cache_folder = "cache"
    graph_folder = "graphs"
    os.makedirs(graph_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)

    models = [
        (classify_with_groq_deepseek, "OpenAI GPT-OSS 120B"),
        (classify_with_groq, "Groq LLaMA3.1 8B"),
        (classify_with_gemini, "Gemini 2.5 Pro"),
        (classify_with_cohere, "Cohere Command-R+"),
        (classify_with_claude, "Claude 3 Haiku"),
        (classify_with_mistral_local, "Mistral Local"),
    ]

    all_binary_results = []
    all_type_results = []

    batch_files = sorted(
        [f for f in os.listdir(batch_folder) if f.endswith(".csv")],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    # Helper for sampling
    def stratified_sample(df_func, df_nfr, n_per_class, seed):
        n_sample = min(n_per_class, len(df_func), len(df_nfr))
        return pd.concat([
            df_func.sample(n=n_sample, random_state=seed),
            df_nfr.sample(n=n_sample, random_state=seed)
        ]).reset_index(drop=True)

    sampling_levels = {
        "sample_10": 5,    # 5 FR + 5 NFR
        "sample_25": 12,   # 12 FR + 12 NFR ≈ 25 total
        "full_batch": None # all FR + all NFR
    }

    for batch_idx, batch_file in enumerate(batch_files, start=1):
        batch_name = f"batch_{batch_idx}"
        df_batch = pd.read_csv(os.path.join(batch_folder, batch_file))
        df_batch["label"] = df_batch["label"].fillna("No").astype(str)
        df_batch["nfr_type"] = df_batch["nfr_type"].fillna("None").astype(str)

        df_func = df_batch[df_batch["label"].str.lower().isin(["no", "functional", "fr"])]
        df_nfr = df_batch[df_batch["label"].str.lower().isin(["yes", "non-functional", "nfr"])]

        for level_name, n_value in sampling_levels.items():
            if n_value is None:
                df_sampled = pd.concat([df_func, df_nfr]).reset_index(drop=True)
            else:
                df_sampled = stratified_sample(df_func, df_nfr, n_value, seed=42)

            print(f"\n📦 Processing {batch_name} ({level_name}): sampled {len(df_sampled)} requirements")

            # Save sampled data for verification
            sample_file = os.path.join(graph_folder, f"{batch_name}_{level_name}_sampled.csv")
            df_sampled.to_csv(sample_file, index=False)
            print(f"📄 Saved sampled data: {sample_file}")

            for model_fn, model_name in models:
                for technique in techniques:
                    desc = f"{model_name} ({technique}) - {batch_name} ({level_name})"
                    binary_result, type_result = evaluate_model(
                        model_fn, model_name, df_sampled,
                        cache_dir=cache_folder,
                        batch_name=f"{batch_name}_{level_name}",
                        technique=technique,
                        tqdm_desc=desc
                    )
                    all_binary_results.append(binary_result)
                    all_type_results.append(type_result)

            # Plot graph per level
            batch_df = pd.DataFrame([r for r in all_binary_results if r["Batch"] == f"{batch_name}_{level_name}"])
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")
            sns.barplot(data=batch_df, x="Model", y="F1", hue="Technique", ci=None)
            plt.xticks(rotation=30, ha="right")
            plt.title(f"F1 Score (Binary) - {batch_name} ({level_name})")
            plt.tight_layout()
            graph_path = os.path.join(graph_folder, f"graph_{batch_name}_{level_name}.png")
            plt.savefig(graph_path, dpi=300)
            plt.close()
            print(f"📊 Graph saved: {graph_path}")

            # Save summary CSV per level
            batch_binary_df = pd.DataFrame([r for r in all_binary_results if r["Batch"] == f"{batch_name}_{level_name}"])
            batch_type_df = pd.DataFrame([r for r in all_type_results if r["Batch"] == f"{batch_name}_{level_name}"])
            batch_summary = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1"],
                "Binary_Avg": batch_binary_df[["Accuracy", "Precision", "Recall", "F1"]].mean().round(3).values,
                "Type_Avg": batch_type_df[["Accuracy", "Precision", "Recall", "F1"]].mean().round(3).values
            })
            summary_file = os.path.join(graph_folder, f"{batch_name}_{level_name}_summary.csv")
            batch_summary.to_csv(summary_file, index=False)
            print(f"📝 Batch summary saved: {summary_file}")

    # Save overall results
    pd.DataFrame(all_binary_results).to_csv("binary_results.csv", index=False)
    pd.DataFrame(all_type_results).to_csv("type_results.csv", index=False)

    combined_df = pd.DataFrame(all_binary_results)
    plt.figure(figsize=(14, 6))
    sns.barplot(data=combined_df, x="Batch", y="F1", hue="Model", ci=None)
    plt.title("Overall Batch-wise Binary F1 Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_folder, "combined_batchwise_binary_f1.png"), dpi=300)
    plt.close()

    print("\n✅ All batches processed successfully!")
    print("📁 Results saved as binary_results.csv and type_results.csv")
    print("📊 Combined graph saved as combined_batchwise_binary_f1.png")
