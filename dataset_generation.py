import random
from datasets import load_dataset, Dataset
import json

# Load datasets
ds1 = load_dataset("parquet", data_files="data/train/explainer_clean_1250.parquet")["train"]
ds2 = load_dataset("parquet", data_files="data/train/magpie_clean_10k.parquet")["train"]

N = 5 # Number of times to repeat each chess example
M = 1250 # Number of non-chess examples to sample

print("Copying and repeating examples...")
repeated_ds1 = []
for example in ds1:
    for _ in range(N):
        repeated_ds1.append({
            "prompt": example["prompt"],
            "completion": example["completion"]
        })

# Sample M examples from ds2
sampled_ds2 = random.sample(list(ds2), 1250)

sampled_ds2 = [{
    "prompt": example["prompt"],
    "completion": example["completion"]
} for example in sampled_ds2]

# Combine datasets
final_dataset = repeated_ds1 + sampled_ds2

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


hf_dataset = Dataset.from_list(final_dataset)
hf_dataset.to_json("data/llamafactory_programmatic_7500.json")



# Define your dataset names and their paths
datasets = {
    "llmchess_programmatic": {
        "file_name": "llamafactory_programmatic_7500.json"
    }
}

# Define the path where you want to save your dataset_info.json
output_path = "data/dataset_info.json"

# Write the datasets information to the file
with open(output_path, "w") as json_file:
    json.dump(datasets, json_file, indent=2)

print(f"Dataset info saved to {output_path}")

