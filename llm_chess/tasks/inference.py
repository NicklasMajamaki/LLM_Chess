import wandb
import argparse
import subprocess

from llm_chess import utils as utils


# Defining defaults here
TASK_MAP = {
    'bestmove': "choose_from_n",
    'worstmove': "choose_from_n",
    'legalmoves': "produce_list",
    'predictmove': "predict_singlemove",
    'blunder_explanations': "synthetic_generation",
    'good_move_explanations': "synthetic_generation",
}

RUNTYPE_SYSPROMPT_MAPPING = {
    'hallucination': 'hallucinations_sysprompt.txt', 
    'reasoning_strategy': 'reasoning_strategies_sysprompt.txt'
}


# ======================================
# Arg parsing
# ======================================
def none_or_int(val):
    return None if val.lower() == "none" else int(val)

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation.")

    # Model information 
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--model_version", type=str, default="llama3", help="Model version being run to ensure correct special tokens used.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1/completions", help="Base URL for the model endpoint")

    # Filenames and dirs
    parser.add_argument("--data_dir", type=str, default="llm_chess/data/cleaned", help="Path to the data directory")
    parser.add_argument("--data_files", nargs="+", default=None, help="List of data files to use (e.g., evals, rejsampling, train data)")
    
    # Various run details 
    parser.add_argument("--experiment_name", type=str, default='my-experiment', help="Give name for experiment for s3 bucket saving organization.")
    parser.add_argument("--run_type", type=str, default='eval', help="Specify which task you're doing (e.g., 'eval', 'rejsampling', 'hallucination', 'reasoning_strategy').")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of samples to pass into vLLM in each batch.")
    parser.add_argument("--max_samples", type=none_or_int, default=None, help="If set to None, use all your data in your --data-files; if set to int, use that as max number of samples to test on.")
    
    # Logging / saving details
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Use wandb for logging")
    parser.add_argument("--print_verbose", default=False, action="store_true", help="Print all outputs.")
    parser.add_argument("--save_verbose", default=False, action="store_true", help="Save all outputs.")

    # Inference hyperparams
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.02)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    return parser.parse_args()



# ======================================
# Main run code
# ======================================
def main():
    args = parse_args()

    # VLLM client
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

    # For cases where we have data with 'correct answers' that we test against (e.g., evaluations, rejection sampling)
    if args.run_type in ["eval", "rejsampling"]:
        evaluator = utils.Evaluator(
            args = args,
            task_map = TASK_MAP
        )
        print(f"Starting {args.run_type}...")
        results = evaluator.evaluate(client, verbose=False, save_verbose=args.save_verbose)
        print(f"Completed {args.run_type}.\n\nFinal Results:\n{results}")
    
    # For cases where we just want to generate data
    if args.run_type in ["generate"]:
        generator = utils.Generator(
            args = args,
            task_map = TASK_MAP
        )
        print(f"Starting {args.run_type}...")
        generator.generate(client, verbose=False, save_verbose=args.save_verbose)
        print(f"Completed {args.run_type}.")

    # For cases where we want the LLM to parse existing responses to extract more nuanced info (e.g., hallucinations, reasoning methods used)
    if args.run_type in ['hallucination', 'reasoning_strategy']:
        llm_parser = utils.LLMParser(
            args = args,
            runtype_mapping = RUNTYPE_SYSPROMPT_MAPPING
        )
        print(f"Starting {args.run_type}...")
        results = llm_parser.evaluate(client, verbose=False, save_verbose=args.save_verbose)
        print(f"Completed {args.run_type}.\n\nFinal Results:\n{results}")

    # Save to s3 bucket
    cmd = f"aws s3 cp {args.data_dir}/saved_data s3://llm-chess/saved_data/{args.experiment_name} --recursive"
    print(f"S3 save command: {cmd}")
    subprocess.run(cmd.split(), check=True)

    # Remove all files after saving
    cmd = f"rm -f {args.data_dir}/saved_data/*"
    print(f"Cleanup command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()