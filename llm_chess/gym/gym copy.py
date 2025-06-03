import os
from llm_chess.utils import vLLMClient, Evaluator
from collections import defaultdict
import argparse
import subprocess
import wandb


def none_or_int(val):
    return None if val.lower() == "none" else int(val)

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation.")

    # Model information 
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--llama_version", type=str, default="llama3", help="Llama version being run to ensure correct special tokens used.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1/completions", help="Base URL for the model endpoint")

    # Filenames and dirs
    parser.add_argument("--data_dir", type=str, default="llm_chess/data/cleaned", help="Path to the data directory")
    parser.add_argument("--data_files", nargs="+", default='gym_data', help="List of data files to use (e.g., evals, rejsampling, train data)")
    
    # Various run details 
    parser.add_argument("--run_type", type=str, default='eval', help="Specify which task you're doing (e.g., 'eval', 'rejsampling').")
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


class ChessEvalGym:
    def __init__(self, args):
        """
        args: argparse args
        """
        self.args = args

        self.base_eval_dir = os.path.join(os.path.dirname(__file__), 'gym_data', 'gym_evals')

        # Baked-in task difficulty map
        self.task_difficulty_map = {
            "bestmove": [2, 3, 4, 5, 6],
            "worstmove": [2, 3, 4, 5, 6],
            "legalmoves": None
        }

        # Hard-coded thresholds
        self.thresholds = {
            "bestmove": 0.8,
            "worstmove": 0.8,
            "legalmoves": 0.9
        }

        self.results = {}

        self.model = vLLMClient(
            model=args.model,
            base_url=args.base_url,
            generation_args={
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "min_p": args.min_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            }
        )

    def evaluate(self):
        """
        Evaluates the model on all tasks and all difficulties (if applicable).
        Returns:
            - A nested dict of scores: {task_name: {difficulty_level: score}}
              For single-difficulty tasks, use difficulty='N/A'
            - A dict of hardest difficulty passed: {task_name: hardest_level}
        """
        all_scores = {}
        hardest_passed = {}

        for task, difficulties in self.task_difficulty_map.items():
            threshold = self.thresholds.get(task, 0.0)
            task_scores = {}
            hardest_level = None

            if difficulties is None:
                # Single file evaluation
                filename = f"{task}_gym_eval.jsonl"
                filepath = os.path.join(self.base_eval_dir, filename)
                self.args.data_files = [filepath]

                evaluator = Evaluator(
                    args=self.args,
                    task_map={task: task},
                    wandb_run=None
                )
                print(f"Evaluating {task} (no difficulty)...")
                result_dicts = evaluator.evaluate(self.model, verbose=False, save_verbose=False)
                score = result_dicts[0].get("score", 0)
                task_scores['N/A'] = score

                if score >= threshold:
                    hardest_level = 'N/A'

            else:
                # Multiple difficulties
                for difficulty in difficulties:
                    filename = f"{task}_gym_eval_{difficulty}provided.jsonl"
                    filepath = os.path.join(self.base_eval_dir, filename)
                    self.args.data_files = [filepath]

                    evaluator = Evaluator(
                        args=self.args,
                        task_map={task: task},
                        wandb_run=None
                    )
                    print(f"Evaluating {task} at difficulty {difficulty}...")
                    result_dicts = evaluator.evaluate(self.model, verbose=False, save_verbose=False)
                    score = result_dicts[0].get("score", 0)
                    task_scores[difficulty] = score

                    if score >= threshold:
                        hardest_level = difficulty

            all_scores[task] = task_scores
            hardest_passed[task] = hardest_level or "failed_all"

        self.results = {"scores": all_scores, "hardest_passed": hardest_passed}
        return self.results



    def train_on_current_level(self):
        """
        For each task, identify the appropriate training level and call run_rl_training.
        """
        for task, difficulties in self.task_difficulty_map.items():
            failure_level = self.results["hardest_passed"].get(task)

            if difficulties is None:
                # Single difficulty task (legalmoves)
                train_level = 'N/A'
                filename = f"{task}_gym_train.jsonl"
            else:
                if failure_level == "failed_all":
                    train_level = difficulties[0]
                else:
                    failed_index = difficulties.index(failure_level)
                    train_index = max(0, failed_index - 1)
                    train_level = difficulties[train_index]

                filename = f"{task}_gym_train_{train_level}provided.jsonl"

            
            train_data_dir = os.path.join(os.path.dirname(__file__), 'gym_data', 'gym_train')
            data_path = os.path.join(train_data_dir, filename)

            print(f"Starting RL training on {task} (level: {train_level}) at {data_path}")

            self.run_rl_training(task, data_path)


    def run_rl_training(self, task, data_path):
        pass


def main():
    args = parse_args()

    # Set up wandb logger
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

    # Initialize ChessEvalGym
    gym = ChessEvalGym(args)

    # Evaluate the model
    print(f"Starting gym evaluation...")
    results = gym.evaluate()
    print(f"Completed gym evaluation.\n\nFinal Results:\n{results}")

    # Save to s3 bucket (optional)
    cmd = f"aws s3 cp {args.data_dir}/saved_data s3://llm-chess/saved_data --recursive"
    print(f"S3 save command: {cmd}")
    subprocess.run(cmd.split())


if __name__ == "__main__":
    main()