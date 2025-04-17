# Main file we'll run from mcli to set up a vllm endpoint and run parquets through it
import time
import wandb
import argparse
import subprocess

import utils

# Defining defaults here
FILENAME_MAP = {
    'bestmove': "choose_from_n",
    'worstmove': "choose_from_n",
    'legalmoves': "produce_list",
    'predictmove': "predict_singlemove",
}
EVAL_FILES = [
    "bestmove_100.parquet",
    "legalmoves_100.parquet",
    "predictmove_100.parquet",
    "worstmove_100.parquet"
]
# Parsing functionality for CLI args
def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation.")

    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--eval_files", nargs="+", default=EVAL_FILES, help="List of evaluation files")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1/llm_chess", help="Base URL for the model endpoint")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Use wandb for logging")

    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.02)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_evals", default=None)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.use_wandb:
        wandb_run = wandb.init(
            config={
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "min_p": args.min_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            }
        )
    else:
        wandb_run = None

    evaluator = utils.Evaluator(
        datafolder_fp=args.data_dir,
        eval_files=args.eval_files,
        filename_map=FILENAME_MAP,
        batch_size=args.batch_size,
        max_evals=args.max_evals,
        wandb_run=wandb_run,
    )

    client = utils.vLLMClient(
        model=args.model,
        base_url=args.base_url,
        generation_args={
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty
        }
    )

    print("Starting Evaluation...")
    results = evaluator.evaluate(client, verbose=True, save_verbose=True)
    print(f"Evaluation Completed. Final Results:\n{results}")

    # Save to s3 bucket
    cmd = f"aws s3 cp {args.data_dir}/saved_data s3://llm-chess/saved_data --recursive"
    print(cmd)
    subprocess.run(cmd.split())

if __name__ == "__main__":
    main()
