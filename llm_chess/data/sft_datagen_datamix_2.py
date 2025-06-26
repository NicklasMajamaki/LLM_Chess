import json
import random
from typing import List
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets, Dataset

from llm_chess.prompts.chat_to_prompt import ChatProcessor



# Main args to adjust
MAX_SAMPLES = 15000
LLAMA_VERSION = "qwen25"    # {'llama4', 'llama3'}
OUTPUT_FOLDER = "llm_chess/data/"
DATA_FOLDER = "llm_chess/data/cleaned/train_data"
DATASET_CONFIG = [
    {
        "name": "magpie",
        "files": ["magpieclean_20k.jsonl"],
        "weight": 0.25
    },
    {
        "name": "chess_explainer",
        "files": ["combined_chessexplainer_5k.jsonl"],
        "weight": 0.2
    },
    {
        "name": "rejsampling",
        "files": ["rejsampling_clean_1630.jsonl", "rejsampling_clean_1988.jsonl", "rejsampling_clean_3269.jsonl"],
        "weight": 0.45
    },
    {
        "name": "synthetic_moves",
        "files": ["syntheticmoves_blunders_1000.jsonl", "syntheticmoves_good_300.jsonl"],
        "weight": 0.1
    },
]

# Using custom dataclass to load in each dataset
@dataclass
class DatasetSource:
    name: str
    file_paths: List[str]
    weight: float

    def load(self):
        datasets = [load_dataset("json", data_files=fp, split="train") for fp in self.file_paths]
        return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
sources = [
    DatasetSource(
        name=cfg["name"],
        file_paths=[f"{DATA_FOLDER}/{fname}" for fname in cfg["files"]],
        weight=cfg["weight"]
    ) for cfg in DATASET_CONFIG
]

# Calculate samples we'll take from each set
all_loaded = [src.load() for src in sources]
max_by_weight = [
    len(ds) / src.weight if src.weight > 0 else float('inf')
    for ds, src in zip(all_loaded, sources)
]
actual_total = int(min(MAX_SAMPLES, *max_by_weight))
samples_per_set = [int(actual_total * src.weight) for src in sources]
samples_per_set[-1] = actual_total - sum(samples_per_set[:-1])
print("Sample counts per dataset:", {s.name: c for s, c in zip(sources, samples_per_set)})

# Random sample from each dataset
chat_processor = ChatProcessor(LLAMA_VERSION)
final_samples = []
for ds, count in zip(all_loaded, samples_per_set):
    picked = random.sample(list(ds), min(count, len(ds))) if count > 0 else []
    for example in picked:
        prompt, response = chat_processor.process_chat(example['chat'])
        final_samples.append({
            "prompt": prompt,
            "completion": response
        })

# Shuffle and save dataset
random.shuffle(final_samples)
print("Saving dataset...")
dataset_filename = f"llamafactory_programmatic_{len(final_samples)}.json"
hf_dataset = Dataset.from_list(final_samples)
hf_dataset.to_json(f"{OUTPUT_FOLDER}/{dataset_filename}")


# Finally write a dataset_info.json file for llamafactory
datasets = {
    "llmchess_programmatic": {
        "file_name": dataset_filename,
        "columns": {
            "prompt": "prompt",
            "response": "completion"
        }
    }
}
with open(f"{OUTPUT_FOLDER}/dataset_info.json", "w") as json_file:
    json.dump(datasets, json_file, indent=2)
print(f"Dataset info saved to {OUTPUT_FOLDER}/dataset_info.json")