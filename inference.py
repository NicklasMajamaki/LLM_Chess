# Main file we'll run from mcli to set up a vllm endpoint and run parquets through it
import argparse
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation.")

    parser.add_argument("--data_dir", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--eval_files", nargs="+", default=["bestmove_50.parquet", "legalmoves_50.parquet", "predictmove_100.parquet", "worstmove_50.parquet"], help="List of evaluation files")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL for the model endpoint")

    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_evals", type=int, default=1)

    return parser.parse_args()

def main():
    args = parse_args()

    FILENAME_MAP = {
        'bestmove': "choose_from_n",
        'worstmove': "choose_from_n",
        'legalmoves': "produce_list",
        'predictmove': "predict_singlemove",
    }

    evaluator = utils.Evaluator(
        args.data_dir,
        args.eval_files,
        FILENAME_MAP,
        batch_size=args.batch_size,
        max_evals=args.max_evals
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
    results = evaluator.evaluate(client, verbose=True)
    print(f"Evaluation Completed. Final Results:\n{results}")

if __name__ == "__main__":
    main()
