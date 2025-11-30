import pandas as pd
import os
from utils.data_utils import detect_columns, normalize_label

# ======================
# Step 1: Load dataset
# ======================
file_path = "requirements_multiclass_original.csv"
df = pd.read_csv(file_path)

# Detect columns
story_col, label_col = detect_columns(df)
if not story_col or not label_col:
    raise ValueError(f"Could not detect user_story or label columns in {file_path}")

# Normalize label column to 'FR'/'NFR'
df['__normalized_label'] = df[label_col].apply(normalize_label)

# ======================
# Step 2: Separate Functional and Non-Functional
# ======================
functional = df[df['__normalized_label'] == 'FR']
non_functional = df[df['__normalized_label'] == 'NFR']

print("Before balancing:")
print("Functional:", len(functional))
print("Non-Functional:", len(non_functional))

# Downsample majority class to match the smaller one
if len(functional) > len(non_functional):
    functional_sampled = functional.sample(len(non_functional), random_state=42)
    ignored_functional = functional.drop(functional_sampled.index)
    non_functional_sampled = non_functional.copy()
    ignored_non_functional = pd.DataFrame(columns=non_functional.columns)  # empty
else:
    non_functional_sampled = non_functional.sample(len(functional), random_state=42)
    ignored_non_functional = non_functional.drop(non_functional_sampled.index)
    functional_sampled = functional.copy()
    ignored_functional = pd.DataFrame(columns=functional.columns)  # empty

# ======================
# Step 3: Combine balanced dataset
# ======================
balanced_df = pd.concat([functional_sampled, non_functional_sampled]).sample(frac=1, random_state=42)
print("\nAfter balancing:")
print(balanced_df['__normalized_label'].value_counts())

# Save the balanced dataset
balanced_path = "balanced_user_stories.csv"
balanced_df.to_csv(balanced_path, index=False)
print(f"\n✅ Balanced dataset saved as: {balanced_path}")

# ======================
# Step 4: Save ignored requirements (during balancing)
# ======================
ignored_df = pd.concat([ignored_functional, ignored_non_functional]).sample(frac=1, random_state=42)
ignored_path = "ignored_user_stories.csv"
ignored_df.to_csv(ignored_path, index=False)
print(f"✅ Ignored requirements saved as: {ignored_path}")
print(f"Number of ignored requirements: {len(ignored_df)}")

# ======================
# Step 5: Split into batches (25 + 25 per batch)
# ======================
batch_size = 50
per_class_per_batch = batch_size // 2  # 25 of each

# Shuffle both classes again
functional_sampled = functional_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
non_functional_sampled = non_functional_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate number of full batches
num_full_batches = min(len(functional_sampled), len(non_functional_sampled)) // per_class_per_batch

# Create output folder
output_folder = "batches"
os.makedirs(output_folder, exist_ok=True)

# Generate full batches
for i in range(num_full_batches):
    func_batch = functional_sampled.iloc[i * per_class_per_batch : (i + 1) * per_class_per_batch]
    non_func_batch = non_functional_sampled.iloc[i * per_class_per_batch : (i + 1) * per_class_per_batch]
    
    batch_df = pd.concat([func_batch, non_func_batch]).sample(frac=1, random_state=42)
    
    batch_file = os.path.join(output_folder, f"batch_{i+1}.csv")
    batch_df.to_csv(batch_file, index=False)

# ======================
# Add last batch with leftovers if any
# ======================
leftover_func = functional_sampled.iloc[num_full_batches * per_class_per_batch :]
leftover_non_func = non_functional_sampled.iloc[num_full_batches * per_class_per_batch :]

if len(leftover_func) > 0 or len(leftover_non_func) > 0:
    batch_df = pd.concat([leftover_func, leftover_non_func]).sample(frac=1, random_state=42)
    batch_file = os.path.join(output_folder, f"batch_{num_full_batches + 1}.csv")
    batch_df.to_csv(batch_file, index=False)
    print(f"Last batch (smaller) saved as: batch_{num_full_batches + 1}.csv ({len(batch_df)} rows)")

total_batches = num_full_batches + (1 if len(leftover_func) > 0 or len(leftover_non_func) > 0 else 0)
print(f"\n✅ All {total_batches} batches saved in the '{output_folder}' folder!")

# ======================
# Step 6 (Optional): Verify first few batches
# ======================
for file in sorted(os.listdir(output_folder))[:3]:
    print("Sample batch file:", file)
