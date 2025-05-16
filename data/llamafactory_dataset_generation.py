import random
from datasets import load_dataset, concatenate_datasets, Dataset
import json

# Specify our filepaths
output_fp = "./"
magpie_files = [
    "magpie_data/magpie_clean_30k.parquet",
]
explainer_files = [
    "explainer_data/explainer_clean_1250.parquet",
    "explainer_data/explainer_clean_100_1558_15.parquet",
    "explainer_data/explanations_0_1000_0104_16.parquet",
    "explainer_data/explanations_1_1000_0330_16.parquet",
    "explainer_data/explanations_2_1000_0557_16.parquet",
    "explainer_data/explanations_3_1000_0826_16.parquet",
]

# Specify sampling ratios (should sum to 1.0)
explainer_ratio = 0.2
magpie_ratio = 0.8


# Load our datasets 
ds1_list = [load_dataset("parquet", data_files=fp)["train"] for fp in explainer_files]
ds2_list = [load_dataset("parquet", data_files=fp)["train"] for fp in magpie_files]
ds1 = concatenate_datasets(ds1_list)
ds2 = concatenate_datasets(ds2_list)

# Determine total number of samples
TOTAL_SAMPLES = 30000  # Set your desired total number of samples here

# Compute the max possible samples for each dataset given the ratio
max_explainer = len(ds1)
max_magpie = len(ds2)

# Find the largest possible sample size that fits the ratio and available data
max_by_explainer = max_explainer / explainer_ratio if explainer_ratio > 0 else float('inf')
max_by_magpie = max_magpie / magpie_ratio if magpie_ratio > 0 else float('inf')

actual_total = int(min(TOTAL_SAMPLES, max_by_explainer, max_by_magpie))
actual_explainer = int(actual_total * explainer_ratio)
actual_magpie = actual_total - actual_explainer  # Ensure sum matches

print(f"Sampling {actual_explainer} from explainer, {actual_magpie} from magpie (target total: {TOTAL_SAMPLES}, actual total: {actual_total})...")

# Sample from explainer and magpie datasets
explainer_samples = random.sample(list(ds1), actual_explainer) if actual_explainer > 0 else []
magpie_samples = random.sample(list(ds2), actual_magpie) if actual_magpie > 0 else []

# Format samples
explainer_samples = [{
    "prompt": example["prompt"],
    "completion": example["completion"]
} for example in explainer_samples]
magpie_samples = [{
    "prompt": example["prompt"],
    "completion": example["completion"]
} for example in magpie_samples]

# Combine and shuffle
final_dataset = explainer_samples + magpie_samples
print("Shuffling dataset...")
random.shuffle(final_dataset)

print("Saving dataset...")
# Convert to HuggingFace Dataset and save to JSON
# Preview a few entries
for i, entry in enumerate(final_dataset[:5]):
    print(f"Example {i+1}:")
    print("Prompt:", entry["prompt"])
    print("Completion:", entry["completion"])
    print("-" * 40)

# Save our final dataset
dataset_filename = f"llamafactory_programmatic_{len(final_dataset)}.json"
hf_dataset = Dataset.from_list(final_dataset)
hf_dataset.to_json(f"{output_fp}/{dataset_filename}")



# Define your dataset names and their paths
datasets = {
    "llmchess_programmatic": {
        "file_name": dataset_filename,
        "columns": {
            "prompt": "prompt",
            "response": "completion"
        }
    }
}

# Write the datasets information to the file
with open(f"{output_fp}/dataset_info.json", "w") as json_file:
    json.dump(datasets, json_file, indent=2)
print(f"Dataset info saved to {output_fp}/dataset_info.json")