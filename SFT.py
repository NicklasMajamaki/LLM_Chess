import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import IterableDataset
from transformers import default_data_collator
from datasets import load_dataset, concatenate_datasets
import random
import subprocess
from torch.utils.data import DataLoader, Dataset



MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

local_output_dir = "./sft-model"
s3_output_dir = "s3://llm-chess/saved_models/sft-model"

ds1 = load_dataset("parquet", data_files="data/train/explainer_clean_1250.parquet")["train"]
ds2 = load_dataset("parquet", data_files="data/train/magpie_clean_10k.parquet")["train"]

# More datasets can be added, but make sure the probabilities sum to 1
datasets = [ds1, ds2]
probs = [0.7, 0.3]

class ProbabilisticPromptCompletionDataset(IterableDataset):
    def __init__(self, datasets, probs, tokenizer, max_length=2048):
        self.datasets = datasets
        self.probs = probs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        while True:
            ds = random.choices(self.datasets, weights=self.probs, k=1)[0]
            example = random.choice(ds)
            prompt = example["prompt"]
            completion = example["completion"]

            # Tokenize prompt and completion
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.max_length)
            remaining_length = self.max_length - len(prompt_tokens["input_ids"])
            completion_tokens = self.tokenizer(completion, add_special_tokens=False, truncation=True, max_length=remaining_length)

            input_ids = prompt_tokens["input_ids"] + completion_tokens["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_tokens["input_ids"]) + completion_tokens["input_ids"]

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }


print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
dataset = ProbabilisticPromptCompletionDataset(datasets, probs, tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# OUTPUT_DIR WILL NEED TO BE CHANGED, SAME WITH SAVE/LOGGING STEPS
# Pretty much every parameter here should be changed, but the framework is here
training_args = TrainingArguments(
    output_dir=local_output_dir,
    bf16=True,
    learning_rate=1e-5,
    max_steps=5000,
    per_device_train_batch_size=8,
    #deepspeed="ds_config_zero2.json",
    #ddp_backend = "nccl",
    lr_scheduler_type="cosine", 
    num_train_epochs=3,
    logging_steps=1,
    save_steps=500,
    save_total_limit=4,
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

cmd = f"aws s3 cp {local_output_dir} {s3_output_dir} --recursive"
print(f"Uploading model to S3 with command:\n{cmd}")
subprocess.run(cmd.split())