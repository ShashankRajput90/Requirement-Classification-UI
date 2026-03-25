import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# FILE PATHS
# ===============================

context_binary_file = "FINAL_context_binary_metrics.csv"
context_type_file = "FINAL_context_type_metrics.csv"

baseline_binary_file = "binary_results.csv"
baseline_type_file = "type_results.csv"

output_folder = "comparison_results"
os.makedirs(output_folder, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================

context_binary = pd.read_csv(context_binary_file)
context_type = pd.read_csv(context_type_file)

baseline_binary = pd.read_csv(baseline_binary_file)
baseline_type = pd.read_csv(baseline_type_file)

# Rename columns for clarity
context_binary["Approach"] = "Context"
baseline_binary["Approach"] = "Baseline"

context_type["Approach"] = "Context"
baseline_type["Approach"] = "Baseline"

# ===============================
# STANDARDIZE COLUMN NAMES
# ===============================

# Ensure both datasets use same column name for F1
if "Binary_F1" in context_binary.columns:
    context_binary.rename(columns={"Binary_F1": "F1"}, inplace=True)

if "Binary_F1_Weighted" in context_binary.columns:
    context_binary.drop(columns=["Binary_F1_Weighted"], inplace=True)

# ===============================
# MERGE BINARY RESULTS
# ===============================

binary_combined = pd.concat([context_binary, baseline_binary], ignore_index=True)

binary_combined.to_csv(
    os.path.join(output_folder, "binary_comparison_full.csv"),
    index=False
)

# ===============================
# MERGE TYPE RESULTS
# ===============================

context_type.rename(columns={"Type_F1": "F1"}, inplace=True)

type_combined = pd.concat([context_type, baseline_type], ignore_index=True)

type_combined.to_csv(
    os.path.join(output_folder, "type_comparison_full.csv"),
    index=False
)

# ===============================
# CALCULATE OVERALL AVERAGE
# ===============================

binary_avg = binary_combined.groupby(["Model", "Technique", "Approach"])["F1"].mean().reset_index()
type_avg = type_combined.groupby(["Model", "Technique", "Approach"])["F1"].mean().reset_index()

binary_avg.to_csv(os.path.join(output_folder, "binary_average_comparison.csv"), index=False)
type_avg.to_csv(os.path.join(output_folder, "type_average_comparison.csv"), index=False)

print("✅ Average comparison files saved.")

# ===============================
# CALCULATE IMPROVEMENT %
# ===============================

improvement_results = []

for model in binary_avg["Model"].unique():
    for technique in binary_avg["Technique"].unique():

        context_score = binary_avg[
            (binary_avg["Model"] == model) &
            (binary_avg["Technique"] == technique) &
            (binary_avg["Approach"] == "Context")
        ]["F1"].values

        baseline_score = binary_avg[
            (binary_avg["Model"] == model) &
            (binary_avg["Technique"] == technique) &
            (binary_avg["Approach"] == "Baseline")
        ]["F1"].values

        if len(context_score) > 0 and len(baseline_score) > 0:
            improvement = context_score[0] - baseline_score[0]
            percent = (improvement / baseline_score[0]) * 100 if baseline_score[0] != 0 else 0

            improvement_results.append({
                "Model": model,
                "Technique": technique,
                "Baseline_F1": round(baseline_score[0], 3),
                "Context_F1": round(context_score[0], 3),
                "Absolute_Improvement": round(improvement, 3),
                "Percentage_Improvement": round(percent, 2)
            })

improvement_df = pd.DataFrame(improvement_results)
improvement_df.to_csv(os.path.join(output_folder, "binary_improvement.csv"), index=False)

print("✅ Improvement analysis saved.")

# ===============================
# VISUALIZATION
# ===============================

plt.figure(figsize=(14, 6))
sns.barplot(data=binary_combined, x="Model", y="F1", hue="Approach")
plt.title("Context vs Baseline - Binary F1 Comparison")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "binary_context_vs_baseline.png"), dpi=300)
plt.close()

plt.figure(figsize=(14, 6))
sns.barplot(data=type_combined, x="Model", y="F1", hue="Approach")
plt.title("Context vs Baseline - Type F1 Comparison")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "type_context_vs_baseline.png"), dpi=300)
plt.close()

print("📊 Graphs saved successfully!")
print("🎯 Comparison Complete!")