import wandb
import subprocess
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

local_output_dir = "./sft-model"
s3_output_dir = "s3://llm-chess/saved_models/sft-model"

# Load datasets
ds1 = load_dataset("parquet", data_files="data/train/explainer_clean_1250.parquet")["train"]
ds2 = load_dataset("parquet", data_files="data/train/magpie_clean_10k.parquet")["train"]

# Weighted upsample: replicate ds1 and ds2 so the combined set is ~40% ds1, 60% ds2
len1, len2 = len(ds1), len(ds2)
total = len1 + len2
target1, target2 = int(0.4 * (len1 + len2)), int(0.6 * (len1 + len2))
ratio1, ratio2 = max(target1 // len1, 1), max(target2 // len2, 1)

ds1_upsampled = ds1.select(np.tile(np.arange(len1), ratio1)[:target1])
ds2_upsampled = ds2.select(np.tile(np.arange(len2), ratio2)[:target2])

# Combine and shuffle
combined = concatenate_datasets([ds1_upsampled, ds2_upsampled]).shuffle(seed=42)

class PromptCompletionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=4096):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = example["prompt"]
        completion = example["completion"]

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        # Limit completion so prompt+completion <= max_length
        prompt_len = len(prompt_tokens["input_ids"])
        remaining_length = self.max_length - prompt_len
        # If prompt already takes max_length, return truncated prompt, empty completion
        if remaining_length <= 0:
            input_ids = prompt_tokens["input_ids"][:self.max_length]
            attention_mask = [1] * self.max_length
            labels = [-100] * self.max_length
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        completion_tokens = self.tokenizer(
            completion,
            add_special_tokens=False,
            truncation=True,
            max_length=remaining_length,
        )

        input_ids = prompt_tokens["input_ids"] + completion_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_tokens["input_ids"]) + completion_tokens["input_ids"]

        # Pad if needed
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Use a real pad token if available, otherwise eos
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = PromptCompletionDataset(combined, tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# W&B init (specify your project and run name)
wandb.init(entity="lucasdino-ucsd", project="sft")

training_args = TrainingArguments(
    output_dir=local_output_dir,
    bf16=True,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    lr_scheduler_type="cosine",
    num_train_epochs=1,       
    logging_steps=10,
    save_strategy="epoch",    
    save_total_limit=1,       
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print("Saving final model and uploading to S3...")
trainer.save_model(local_output_dir)

cmd = f"aws s3 cp {local_output_dir} {s3_output_dir} --recursive"
print(f"Uploading model to S3 with command:\n{cmd}")
subprocess.run(cmd.split())