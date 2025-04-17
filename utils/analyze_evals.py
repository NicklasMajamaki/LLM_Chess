import os
import ast
import time
import json
import asyncio
import pandas as pd

from .exceptions import ParseException, IllegalMoveException
from .parsing import extract_solution, coerce_response



class EvaluationDataframe():
    def __init__(self, datafolder, filename, filename_map):
        """ Load the dataset and store useful metadata associated with it. """
        self.filename = filename
        self.datafolder = datafolder
        self.filepath = os.path.join(datafolder, "evals", filename)
        self.eval_type = next(v for k, v in filename_map.items() if filename.startswith(k))
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
    def __init__(self, eval_type, filename, wandb_run):
        self.filename = filename
        self.trimmed_filename = filename.rsplit("_", 1)[0]
        self.wandb_run = wandb_run
        self.eval_type = eval_type
        self.results = self._instantiate_dict()

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

    def add_result(self, model_response, ground_truth):
        try:
            self.results["Total Samples"] += 1
            predicted_answer = coerce_response(extract_solution(model_response), self.eval_type)
            if self.eval_type == "choose_from_n":
                answer, provided_moves = ground_truth

                # Print for checking performance
                # print(f"Predicted: {predicted_answer}, Answer: {answer}, Correct? {predicted_answer == answer}")

                self.results["Correct"] += int(predicted_answer == answer)
                if predicted_answer in provided_moves and predicted_answer != answer:
                    self.results["Incorrect"] += 1
                else:
                    raise IllegalMoveException("Predicted move is not in the provided moves.")
            elif self.eval_type == 'produce_list':   # We know that 'predicted_answer' will be a list
                answer = ground_truth
                self.results["Total Ground Truth Legal Moves"] += len(answer)

                num_right = 0
                already_guessed = set()
                for move in predicted_answer:
                    if move in answer and move not in already_guessed:
                        already_guessed.add(move)
                        num_right += 1
                        self.results["Predicted Ground Truth Legal Moves"] += 1
                    else:
                        self.results["Illegal Moves"] += 1
                
                # Print for checking performance
                # print(f"Predicted: {predicted_answer}, Answer: {answer} -- got {num_right} correct")
                
            elif self.eval_type == 'predict_singlemove':
                answer_dict = ground_truth
                
                if predicted_answer in answer_dict:
                    self.results["Legal Moves Provided"] += 1
                    sorted_moves = sorted(answer_dict.items(), key=lambda x: x[1])
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_moves) if move == predicted_answer)
                    self.results["Cumulative Rank of Moves Provided"] += predicted_move_idx/len(sorted_moves)
                else:
                    # Print for checking performance
                    formatted_answer = {k: round(v, 3) if isinstance(v, float) else v for k, v in answer_dict.items()}
                    print(f"Predicted: {predicted_answer}, Answer: {formatted_answer} -- answer not in legal moves")
                    
                    raise IllegalMoveException("Predicted move is not in the legal moves.")
                
                # Print for checking performance
                formatted_answer = {k: round(v, 3) if isinstance(v, float) else v for k, v in answer_dict.items()}
                print(f"Predicted: {predicted_answer}, Answer: {formatted_answer} -- got rank {predicted_move_idx}/{len(sorted_moves)}")
                
        except Exception as e:
            if isinstance(e, ParseException):
                self.results["Error: Parsing"] += 1
            elif isinstance(e, IllegalMoveException):
                self.results["Error: Illegal Move"] += 1
            else:
                self.results["Error: Other"] += 1
        
    def get_final_dict(self):
        def safe_div(x, y, default=0): return x / y if y else default
    
        if self.eval_type == "choose_from_n":
            total = self.results["Total Samples"]
            self.results["Accuracy"] = safe_div(self.results["Correct"], total)
            self.results["Error Rate"] = safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"Eval - {self.trimmed_filename}/Accuracy": self.results["Accuracy"],
                    f"Eval - {self.trimmed_filename}/Error Rate": self.results["Error Rate"],
                })

        elif self.eval_type == "produce_list":
            gt_total = self.results["Total Ground Truth Legal Moves"]
            illegal = self.results["Illegal Moves"]
            total = self.results["Total Samples"]
            self.results["Percent Legal Moves Predicted"] = safe_div(self.results["Predicted Ground Truth Legal Moves"], gt_total)
            self.results["Ratio of Legal to Illegal Moves"] = safe_div(self.results["Predicted Ground Truth Legal Moves"], illegal)
            self.results["Error Rate"] = safe_div(self.results['Error: Parsing'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"Eval - {self.trimmed_filename}/Percent Legal Moves Predicted": self.results["Percent Legal Moves Predicted"],
                    f"Eval - {self.trimmed_filename}/Ratio of Legal to Illegal Moves": self.results["Ratio of Legal to Illegal Moves"],
                    f"Eval - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        elif self.eval_type == "predict_singlemove":
            legal = self.results["Legal Moves Provided"]
            total = self.results["Total Samples"]
            self.results["Avg. Rank of Move Provided"] = safe_div(self.results["Cumulative Rank of Moves Provided"], legal, 1)
            self.results["Error Rate"] = safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"Eval - {self.trimmed_filename}/Avg. Rank of Move Provided": self.results["Avg. Rank of Move Provided"],
                    f"Eval - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        return self.results



class Evaluator():
    def __init__(self, datafolder_fp, eval_files, filename_map, batch_size, max_evals, wandb_run):
        """ Given a set of eval_files instantiate an evaluator object to analyze the evals. """
        self.eval_files = eval_files
        self.wandb_run = wandb_run
        self.eval_dfs = [EvaluationDataframe(datafolder_fp, f, filename_map) for f in eval_files]
        self.max_evals = max_evals
        self.batch_size = batch_size

    def evaluate(self, model, verbose=False, save_verbose=False):
        """ 
        Evaluate the model on the eval files. 
        model: The vLLM model to evaluate.
        """
        result_dicts = []

        for eval_df in self.eval_dfs:
            verbose_generations = []
            filename_no_ext = os.path.splitext(eval_df.filename)[0]

            print(f"{'='*50}\n Evaluating: {filename_no_ext}\n{'='*50}")
            df = eval_df.df
            results = ResultsDict(eval_df.eval_type, filename_no_ext, self.wandb_run)

            max_len = len(df) if self.max_evals is None else min(len(df), self.max_evals)
            for start_idx in range(0, max_len, self.batch_size):
                batch_df = df.iloc[start_idx:min(start_idx+self.batch_size, max_len)]
                prompts = [row['prompt'] for _, row in batch_df.iterrows()]
                batch_responses = asyncio.run(model.chat(prompts))

                for (index, row), result in zip(batch_df.iterrows(), batch_responses):
                    
                    results.add_result(result, row['answer'])
                    
                    if verbose:
                        print(f"{'-'*10}\nPrompt:\n{row['prompt']}\n")
                        print(f"Model Response:\n{result}\nGround Truth Answer:\n'{row['answer']}'\n")
                    if save_verbose:
                        verbose_generations.append({
                            "prompt": row['prompt'],
                            "model_response": result,
                            "ground_truth": str(row['answer'])
                        })

            result_dicts.append(results.get_final_dict())
            
            print(f"{'-'*50}\nResults for {eval_df.filename}:")
            for key, value in result_dicts[-1].items():
                print(f"{key}: {value}")
            print(f"{'-'*50}\n\n")

            if save_verbose:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                os.makedirs(os.path.join(eval_df.datafolder, 'saved_data'), exist_ok=True)
                save_path = os.path.join(eval_df.datafolder, 'saved_data', f"{filename_no_ext}_{timestamp}.json")
                with open(save_path, 'w') as f:
                    json.dump(verbose_generations, f, indent=4)

        return result_dicts