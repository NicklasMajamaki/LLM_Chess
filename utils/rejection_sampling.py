import os
import ast
import time
import json
import asyncio
import pandas as pd

from .exceptions import ParseException, IllegalMoveException
from .parsing import extract_solution, coerce_response


class RejectionSamplingDataframe():
    def __init__(self, datafolder, filename, filename_map):
        """ Load the dataset and store useful metadata associated with it. """
        self.filename = filename
        self.datafolder = datafolder
        self.filepath = os.path.join(datafolder, "sampling", filename)
        self.rej_type = next(v for k, v in filename_map.items() if filename.startswith(k))
        self.df = self._load_parquets(self.filepath)

    def _load_parquets(self, filename):
        """ Load the parquet file and return the dataframe. """
        df = pd.read_parquet(filename)

        try:
            # Try to coerce strings in the answer column to lists / dicts
            df['answer'] = df['answer'].apply(ast.literal_eval)
        except (ValueError, SyntaxError) as _:
            pass
        
        print(f"Loaded {filename} with {len(df)} rows and {len(df.columns)} columns. Answer column type: {df['answer'].dtype}")
        return df


class ResultsDict():
    def __init__(self, rej_type, filename, wandb_run):
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.rej_type = rej_type
        self.results = {
            "Filename": self.filename,
            "Total Samples": 0,
            "Correct": 0,
            "Incorrect": 0,
            "Error: Parsing": 0,
            "Error: Other": 0,
        }
        self.correct_responses = []
        self.rejected_responses = []

    def add_result(self, model_response, ground_truth, prompt):
        try:
            self.results["Total Samples"] += 1
            predicted_answer = coerce_response(extract_solution(model_response), self.rej_type)
            
            # Ground truth is expected to be a list of valid moves
            valid_moves = ground_truth if isinstance(ground_truth, list) else [ground_truth]
            
            # Check if the predicted answer is in the list of valid moves
            if predicted_answer in valid_moves:
                self.results["Correct"] += 1
                self.correct_responses.append({
                    "prompt": prompt,
                    "model_response": model_response,
                    "predicted_move": predicted_answer,
                    "valid_moves": valid_moves
                })
            else:
                self.results["Incorrect"] += 1
                self.rejected_responses.append({
                    "prompt": prompt,
                    "model_response": model_response,
                    "predicted_move": predicted_answer,
                    "valid_moves": valid_moves
                })
                
        except ParseException:
            self.results["Error: Parsing"] += 1
            self.rejected_responses.append({
                "prompt": prompt,
                "model_response": model_response,
                "error": "Parsing Error",
                "valid_moves": ground_truth
            })
        except Exception as e:
            self.results["Error: Other"] += 1
            self.rejected_responses.append({
                "prompt": prompt,
                "model_response": model_response,
                "error": str(e),
                "valid_moves": ground_truth
            })
        
    def get_final_dict(self):
        def safe_div(x, y, default=0): return x / y if y else default
        
        total = self.results["Total Samples"]
        self.results["Accuracy"] = safe_div(self.results["Correct"], total)
        
        if self.wandb_run:
            self.wandb_run.log({
                f"Rejection Sampling - {self.trimmed_filename}/Accuracy": self.results["Accuracy"],
            })

        return self.results

    def save_responses(self, datafolder):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(datafolder, 'saved_data'), exist_ok=True)
        
        # Save correct responses
        correct_count = len(self.correct_responses)
        if correct_count > 0:
            correct_path = os.path.join(
                datafolder, 
                'saved_data', 
                f"correctresponses_{self.trimmed_filename}_{timestamp}_{correct_count}.json"
            )
            with open(correct_path, 'w') as f:
                json.dump(self.correct_responses, f, indent=4)
            print(f"Saved {correct_count} correct responses to {correct_path}")
            
        # Save rejected responses
        rejected_count = len(self.rejected_responses)
        if rejected_count > 0:
            rejected_path = os.path.join(
                datafolder, 
                'saved_data', 
                f"rejectedresponses_{self.trimmed_filename}_{timestamp}_{rejected_count}.json"
            )
            with open(rejected_path, 'w') as f:
                json.dump(self.rejected_responses, f, indent=4)
            print(f"Saved {rejected_count} rejected responses to {rejected_path}")


class RejectionSampler():
    def __init__(self, datafolder_fp, rej_files, filename_map, batch_size, max_rejs, wandb_run):
        """ Given a set of rej_files instantiate a rejection sampler to filter responses. """
        self.rej_files = rej_files
        self.wandb_run = wandb_run
        self.datafolder = datafolder_fp
        self.rej_dfs = [RejectionSamplingDataframe(datafolder_fp, f, filename_map) for f in rej_files]
        self.max_rejs = max_rejs
        self.batch_size = batch_size

    def sample(self, model, verbose=False):
        """ 
        Run rejection sampling on the model responses based on the rej files.
        model: The vLLM model to evaluate.
        """
        result_dicts = []

        for rej_df in self.rej_dfs:
            filename_no_ext = os.path.splitext(rej_df.filename)[0]

            print(f"{'='*50}\n Rejection Sampling: {filename_no_ext}\n{'='*50}")
            df = rej_df.df
            results = ResultsDict(rej_df.rej_type, filename_no_ext, self.wandb_run)

            max_len = len(df) if self.max_rejs is None else min(len(df), self.max_rejs)
            for start_idx in range(0, max_len, self.batch_size):
                batch_df = df.iloc[start_idx:min(start_idx+self.batch_size, max_len)]
                prompts = [row['prompt'] for _, row in batch_df.iterrows()]
                batch_responses = asyncio.run(model.chat(prompts))

                for (_, row), result in zip(batch_df.iterrows(), batch_responses):
                    results.add_result(result, row['answer'], row['prompt'])
                    
                    if verbose:
                        print(f"{'-'*10}\nPrompt:\n{row['prompt']}\n")
                        print(f"Model Response:\n{result}\nGround Truth Answer:\n'{row['answer']}'\n")

            # Save the responses to files
            results.save_responses(self.datafolder)
            
            result_dict = results.get_final_dict()
            result_dicts.append(result_dict)
            
            print(f"{'-'*50}\nResults for {rej_df.filename}:")
            print(f"Total Samples: {result_dict['Total Samples']}")
            print(f"Correct: {result_dict['Correct']} ({result_dict['Accuracy']*100:.2f}%)")
            print(f"Incorrect: {result_dict['Incorrect']}")
            print(f"Errors: {result_dict['Error: Parsing'] + result_dict['Error: Other']}")
            print(f"{'-'*50}\n\n")

        return result_dicts