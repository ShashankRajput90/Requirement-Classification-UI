import pandas as pd
import re
import json
import os
import hashlib
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
from tqdm import tqdm

# Load API keys from .env
load_dotenv()

# ====================
# Import your LLM functions
# ====================
from code_integration import (
    classify_with_groq_deepseek,
    classify_with_groq,
    classify_with_gemini,
    classify_with_cohere,
    classify_with_claude,
    classify_with_mistral_local
)

# ====================
# Techniques
# ====================
techniques = ["zero_shot", "few_shot", "chain_of_thought", "role_based", "react"]

# ====================
# Helpers for extraction
# ====================
def extract_is_nfr(response: str) -> str:
    if not response:
        return "No"
    match = re.search(r"1\. Is NFR:\s*(Yes|No)", response, re.IGNORECASE)
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"
    if "Non-Functional" in response or "Category: Non-Functional" in response:
        return "Yes"
    return "No"

def extract_nfr_type_multilabel(response: str):
    if not response:
        return ["Unknown"]

    match = re.search(r"2\. NFR Type:\s*(.*)", response, re.IGNORECASE)
    types_str = match.group(1) if match else response

    keyword_map = {
        "Performance": ["performance", "speed", "latency", "response time", "throughput"],
        "Security": ["security", "secure", "encryption", "authentication", "authorization"],
        "Usability": ["usability", "user-friendly", "ease of use", "ux"],
        "Reliability": ["reliability", "reliable", "robustness", "robust"],
        "Maintainability": ["maintainability", "maintainable", "modifiability", "modularity", "extensibility"],
        "Scalability": ["scalability", "scalable"],
        "Availability": ["availability", "uptime", "fault tolerance", "redundancy"],
        "Accessibility": ["accessibility", "a11y", "inclusive design"],
        "Interoperability": ["interoperability", "compatibility", "integration"],
        "Portability": ["portability", "portable", "cross-platform"],
        "Quality": ["quality"],
        "Testability": ["testability", "testable"],
        "Compliance": ["compliance", "regulation", "standard"],
        "Accuracy": ["accuracy", "preciseness", "precision"],
        "Safety": ["safety", "safe"],
        "Efficiency": ["efficiency", "efficient", "resource usage"],
        "Adaptability": ["adaptability", "adaptive", "flexibility"],
        "Robustness": ["robustness", "robust"],
    }

    detected = []
    for label, keywords in keyword_map.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", types_str, re.IGNORECASE):
                detected.append(label)
                break

    return detected if detected else ["Unknown"]

# ====================
# Cache helper
# ====================
def get_cache_key(story: str) -> str:
    return hashlib.md5(story.encode()).hexdigest()

# ====================
# Evaluation function
# ====================
def evaluate_model(model_fn, model_name, df, max_workers=5, technique="few_shot"):
    cache_file = f"cache_{model_name.replace(' ', '_')}.json"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    y_true_bin = df["label"].values
    y_true_type = df["nfr_type"].apply(
        lambda x: [t.strip() for t in str(x).replace(",", "-").split("-") if t.strip() != ""]
    )

    y_pred_bin = [None] * len(df)
    y_pred_type = [None] * len(df)

    def get_response(story):
        key = get_cache_key(story)
        if key in cache:
            cached_resp = cache[key]
            if not str(cached_resp).startswith("❌") and "Unknown" not in str(cached_resp):
                return cached_resp

        try:
            response = model_fn(story, technique)
            cache[key] = response
            return response
        except Exception as e:
            error_msg = f"❌ {model_name} Error: {e}"
            print(error_msg)
            cache[key] = error_msg
            return error_msg

    if model_name.lower().startswith("claude"):
        for i, story in enumerate(tqdm(df["story"], desc=f"Evaluating {model_name}")):
            response = get_response(story)
            y_pred_bin[i] = extract_is_nfr(response)
            y_pred_type[i] = extract_nfr_type_multilabel(response)
            time.sleep(12)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(get_response, story): i for i, story in enumerate(df["story"])}
            for future in tqdm(as_completed(future_to_idx), total=len(df), desc=f"Evaluating {model_name}"):
                idx = future_to_idx[future]
                try:
                    response = future.result()
                    y_pred_bin[idx] = extract_is_nfr(response)
                    y_pred_type[idx] = extract_nfr_type_multilabel(response)
                except Exception as e:
                    print(f"⚠️ Error at index {idx} for {model_name}: {e}")
                    y_pred_bin[idx] = "Error"
                    y_pred_type[idx] = ["Unknown"]

    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    # Binary metrics
    filtered_true_bin = [yt for yt, yp in zip(y_true_bin, y_pred_bin) if yp not in ["Error", None]]
    filtered_pred_bin = [yp for yp in y_pred_bin if yp not in ["Error", None]]

    filtered_true_bin = ["Yes" if str(v).lower() in ["yes", "non-functional"] else "No" for v in filtered_true_bin]
    filtered_pred_bin = ["Yes" if str(v).lower() == "yes" else "No" for v in filtered_pred_bin]

    if not filtered_true_bin:
        bin_results = {"Model": model_name, "Accuracy": None, "Precision (NFR)": None, "Recall (NFR)": None, "F1 (NFR)": None}
    else:
        acc = accuracy_score(filtered_true_bin, filtered_pred_bin)
        prec = precision_score(filtered_true_bin, filtered_pred_bin, pos_label="Yes", zero_division=0)
        rec = recall_score(filtered_true_bin, filtered_pred_bin, pos_label="Yes", zero_division=0)
        f1 = f1_score(filtered_true_bin, filtered_pred_bin, pos_label="Yes", zero_division=0)
        bin_results = {"Model": model_name, "Accuracy": round(acc, 3), "Precision (NFR)": round(prec, 3),
                       "Recall (NFR)": round(rec, 3), "F1 (NFR)": round(f1, 3)}

    # Multi-label metrics
    mlb = MultiLabelBinarizer()
    y_true_bin_type = mlb.fit_transform(y_true_type)
    y_pred_cleaned = [[lab for lab in labels if lab in mlb.classes_] or ["Unknown"] for labels in y_pred_type]
    y_pred_bin_type = mlb.transform(y_pred_cleaned)

    report = classification_report(
        y_true_bin_type,
        y_pred_bin_type,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True
    )

    acc_type = report.get("accuracy", accuracy_score(y_true_bin_type, y_pred_bin_type))

    type_results = {
        "Model": model_name,
        "Accuracy": round(acc_type, 3),
        "Macro Precision": round(report["macro avg"]["precision"], 3),
        "Macro Recall": round(report["macro avg"]["recall"], 3),
        "Macro F1": round(report["macro avg"]["f1-score"], 3),
        "Weighted Precision": round(report["weighted avg"]["precision"], 3),
        "Weighted Recall": round(report["weighted avg"]["recall"], 3),
        "Weighted F1": round(report["weighted avg"]["f1-score"], 3)
    }

    for label, metrics in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            type_results[f"Precision ({label})"] = round(metrics["precision"], 3)
            type_results[f"Recall ({label})"] = round(metrics["recall"], 3)
            type_results[f"F1 ({label})"] = round(metrics["f1-score"], 3)

    # Save per-story predictions
    predictions_df = pd.DataFrame({
        "Id": df.index,
        "Story": df["story"],
        "Ground Truth Label": y_true_bin,
        "Predicted Label": y_pred_bin,
        "Ground Truth Type(s)": y_true_type,
        "Predicted Type(s) (Improved)": y_pred_type,
        "Raw Response": [cache[get_cache_key(story)] for story in df["story"]]
    })
    predictions_filename = f"predictions_{model_name.replace(' ', '_')}.csv"
    predictions_df.to_csv(predictions_filename, index=False)
    print(f"📄 Saved detailed predictions to {predictions_filename}")

    return bin_results, type_results

# ====================
# Main
# ====================
if __name__ == "__main__":
    df = pd.read_csv("batch_3.csv")
    df["nfr_type"] = df["nfr_type"].fillna("None").astype(str)

    models = [
        (classify_with_mistral_local, "Mistral Local"),
        (classify_with_groq, "Groq LLaMA3"),
        (classify_with_gemini, "Gemini"),
        (classify_with_cohere, "Cohere"),
        (classify_with_claude, "Claude"),
        (classify_with_mistral_local, "Mistral Local"),
    ]

    results_binary = []
    results_types = []

    for fn, model_name in models:
        for technique in techniques:
            combined_name = f"{model_name} ({technique})"
            print(f"\n🔹 Evaluating {combined_name} ...")
            bin_metrics, type_metrics = evaluate_model(fn, combined_name, df, max_workers=5, technique=technique)
            results_binary.append(bin_metrics)
            if type_metrics:
                results_types.append(type_metrics)

    # Save Binary Results
    results_binary_df = pd.DataFrame(results_binary)
    results_binary_df.to_csv("nfr_binary_results.csv", index=False)
    print("\n✅ Binary evaluation completed! Results saved in nfr_binary_results.csv")
    print(results_binary_df)

    # Save Type Results
    if results_types:
        results_types_df = pd.DataFrame(results_types)
        results_types_df.to_csv("nfr_type_results.csv", index=False)
        print("\n✅ NFR type evaluation completed! Results saved in nfr_type_results.csv")
        print(results_types_df)

    # ====================
    # Print Best Metrics Summary (Binary)
    # ====================
    if not results_binary_df.empty:
        best_accuracy_row = results_binary_df.loc[results_binary_df["Accuracy"].idxmax()]
        best_precision_row = results_binary_df.loc[results_binary_df["Precision (NFR)"].idxmax()]
        best_recall_row = results_binary_df.loc[results_binary_df["Recall (NFR)"].idxmax()]
        best_f1_row = results_binary_df.loc[results_binary_df["F1 (NFR)"].idxmax()]

        print("\n🏆 Best Binary Metrics Summary:")
        print(f"🏆 Best Accuracy: {best_accuracy_row['Model']} with {best_accuracy_row['Accuracy']}")
        print(f"🏆 Best Precision (NFR): {best_precision_row['Model']} with {best_precision_row['Precision (NFR)']}")
        print(f"🏆 Best Recall (NFR): {best_recall_row['Model']} with {best_recall_row['Recall (NFR)']}")
        print(f"🏆 Best F1 (NFR): {best_f1_row['Model']} with {best_f1_row['F1 (NFR)']}")

    # ====================
    # Print Best Metrics Summary (NFR Type)
    # ====================
    if results_types:
        type_metrics_df = pd.DataFrame(results_types)
        best_macro_f1_row = type_metrics_df.loc[type_metrics_df["Macro F1"].idxmax()]
        best_weighted_f1_row = type_metrics_df.loc[type_metrics_df["Weighted F1"].idxmax()]

        print("\n🏆 Best NFR Type Metrics Summary:")
        print(f"🏆 Best Macro F1: {best_macro_f1_row['Model']} with {best_macro_f1_row['Macro F1']}")
        print(f"🏆 Best Weighted F1: {best_weighted_f1_row['Model']} with {best_weighted_f1_row['Weighted F1']}")
