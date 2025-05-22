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
        self.filepath = os.path.join(datafolder, "rej_sampling", filename)
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
        self.trimmed_filename = filename.split("_", 2)[0]
        self.wandb_run = wandb_run
        self.rej_type = rej_type
        self.results = self._instantiate_dict()

        self.correct_responses = []
        self.rejected_responses = []

    def _instantiate_dict(self):
        if self.eval_type == "choose_from_n":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Correct": 0,
                "Incorrect": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        elif self.eval_type == "produce_list":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Total Ground Truth Legal Moves": 0,
                "Predicted Ground Truth Legal Moves": 0,
                "Illegal Moves": 0,
                "Error: Parsing": 0,
                "Error: Other": 0,
            }
        elif self.eval_type == "predict_singlemove":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Legal Moves Provided": 0,
                "Cumulative Rank of Moves Provided": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        else:
            raise ValueError(f"Undefined eval type: {self.eval_type}")

    def add_result(self, model_response, ground_truth, prompt):
        try:
            self.results["Total Samples"] += 1
            if self.eval_type == "choose_from_n":
                predicted_answer = coerce_response(extract_solution(model_response), self.eval_type)
                answer, provided_moves = ground_truth

                if predicted_answer == answer:
                    self.results["Correct"] += 1    
                    self.correct_responses.append(model_response)
                else:
                    if predicted_answer in provided_moves:
                        self.results["Incorrect"] += 1
                    else:
                        raise IllegalMoveException("Predicted move is not in the provided moves.")
            elif self.eval_type == 'produce_list':   # We know that 'predicted_answer' will be a list
                answer = ground_truth
                self.results["Total Ground Truth Legal Moves"] += len(answer)
                predicted_answer = coerce_response(extract_solution(model_response), self.eval_type)

                num_right = 0
                already_guessed = set()
                for move in predicted_answer:
                    if move in answer and move not in already_guessed:
                        already_guessed.add(move)
                        num_right += 1
                        self.results["Predicted Ground Truth Legal Moves"] += 1
                    else:
                        self.results["Illegal Moves"] += 1
                
                if already_guessed == set(answer):
                    self.correct_responses.append({
                        "prompt": prompt,
                        "completion": model_response
                    })
                
            elif self.eval_type == 'predict_singlemove':
                predicted_answer = coerce_response(extract_solution(model_response), self.eval_type)
                answer_dict = ground_truth
                
                if predicted_answer in answer_dict:
                    self.results["Legal Moves Provided"] += 1
                    sorted_moves = sorted(answer_dict.items(), key=lambda x: x[1])
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_moves) if move == predicted_answer)
                    self.results["Cumulative Rank of Moves Provided"] += predicted_move_idx/len(sorted_moves)

                    # Sorted in increasing order
                    if predicted_move_idx / len(sorted_moves) > 0.5:
                        self.correct_responses.append(model_response)
                else:
                    raise IllegalMoveException("Predicted move is not in the legal moves.")
                                
        except Exception as e:
            if isinstance(e, ParseException):
                self.results["Error: Parsing"] += 1
            elif isinstance(e, IllegalMoveException):
                self.results["Error: Illegal Move"] += 1
            else:
                self.results["Error: Other"] += 1

        
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