import os
import ast
import time
import json
import pandas as pd

from exceptions import ParseException, IllegalMoveException
from utils.parsing import extract_solution, coerce_response



class EvaluationDataframe():
    def __init__(self, datafolder, filename, filename_map):
        """ Load the dataset and store useful metadata associated with it. """
        self.filename = filename
        self.datafolder = datafolder
        self.filepath = os.joinpath(datafolder, "evals", filename)
        self.eval_type = next(v for k, v in filename_map.items() if filename.startswith(k))
        self.df = self._load_parquets(self.filepath)

    def _load_parquets(filename):
        """ Load the parquet file and return the dataframe. """
        df = pd.read_parquet(filename)

        try:
            # Try to coerce strings in the answer column to lists / dicts
            df['answer'] = df['answer'].apply(ast.literal_eval)
        except (ValueError, SyntaxError) as _:
            pass
        
        print(f"Loaded {filename} with {len(df)} rows and {len(df.columns)} columns. Answer column type: {df['answer'].dtype}")
        for col in df.columns:
            print(f"Col {col} is of type: {df[col].dtype}")

        return df


class ResultsDict():
    def __init__(self, eval_type):
        self.eval_type = eval_type
        self.results = self._instantiate_dict()

    def _instantiate_dict(self):
        if self.eval_type == "choose_from_n":
            return {
                "Total Samples": 0,
                "Correct": 0,
                "Incorrect": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        elif self.eval_type == "produce_list":
            return {
                "Total Samples": 0,
                "Total Ground Truth Legal Moves": 0,
                "Predicted Ground Truth Legal Moves": 0,
                "Illegal Moves": 0,
                "Error: Parsing": 0,
                "Error: Other": 0,
            }
        elif self.eval_type == "predict_singlemove":
            return {
                "Total Samples": 0,
                "Legal Moves Provided": 0,
                "Avg. Rank of Move Provided": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        else:
            raise ValueError(f"Undefined eval type: {self.eval_type}")

    def add_result(self, model_response, ground_truth):
        try:
            predicted_answer = coerce_response(extract_solution(model_response), self.eval_type)
            if self.eval_type == "choose_from_n":
                answer, provided_moves = ground_truth
                self.results["Total Samples"] += 1

                # Print for checking performance
                print(f"Predicted: {predicted_answer}, Answer: {answer}, Correct? {predicted_answer == answer}")

                self.results["Correct"] += int(predicted_answer == answer)
                if predicted_answer in provided_moves:
                    self.results["Incorrect"] += 1
                else:
                    raise IllegalMoveException("Predicted move is not in the provided moves.")
            elif self.eval_type == 'produce_list':   # We know that 'predicted_answer' will be a list
                answer = ground_truth
                self.results["Total Samples"] += 1
                self.results["Total Ground Truth Legal Moves"] += len(answer)

                num_right = 0
                for move in predicted_answer:
                    if move in answer:
                        num_right += 1
                        self.results["Predicted Ground Truth Legal Moves"] += 1
                    else:
                        self.results["Illegal Moves"] += 1
                
                # Print for checking performance
                print(f"Predicted: {predicted_answer}, Answer: {answer} -- got {num_right} correct")
                
            elif self.eval_type == 'predict_singlemove':
                answer_dict = ground_truth
                self.results["Total Samples"] += 1
                
                if predicted_answer in answer_dict['legal_moves']:
                    self.results["Legal Moves Provided"] += 1
                    sorted_moves = sorted(answer_dict['legal_moves'].items(), key=lambda x: x[1])
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_moves) if move == predicted_answer)
                    self.results["Cumulative Rank of Moves Provided"] += predicted_move_idx/len(sorted_moves)
                else:
                    # Print for checking performance
                    print(f"Predicted: {predicted_answer}, Answer: {answer_dict['legal_moves']} -- answer not in legal moves")
                    
                    raise IllegalMoveException("Predicted move is not in the legal moves.")
                
                # Print for checking performance
                print(f"Predicted: {predicted_answer}, Answer: {answer_dict['legal_moves']} -- got rank {predicted_move_idx}/{len(sorted_moves)}")
                
        except Exception as e:
            if e is ParseException:
                self.results["Error: Parsing"] += 1
            elif e is IllegalMoveException:
                self.results["Error: Illegal Move"] += 1
            else:
                self.results["Error: Other"] += 1
                
    def get_final_dict(self):
        if self.eval_type == "choose_from_n":
            self.results["Accuracy"] = self.results["Correct"] / self.results["Total Samples"]
            self.results["Error Rate"] = (self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other']) / self.results["Total Samples"]
        elif self.eval_type == "produce_list":
            self.results["Percent Legal Moves Predicted"] = self.results["Predicted Ground Truth Legal Moves"] / self.results["Total Ground Truth Legal Moves"]
            self.results["Ratio of Legal to Illegal Moves"] = self.results["Predicted Ground Truth Legal Moves"] / self.results["Illegal Moves"]
            self.results["Error Rate"] = (self.results['Error: Parsing'] + self.results['Error: Other']) / self.results["Total Samples"]
        elif self.eval_type == "predict_singlemove":
            self.results["Avg. Rank of Move Provided"] = self.results["Cumulative Rank of Moves Provided"] / self.results["Legal Moves Provided"]
            self.results["Error Rate"] = (self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other']) / self.results["Total Samples"]
        
        return self.results


class Evaluator():
    def __init__(self, datafolder_fp, eval_files, filename_map, batch_size=4, max_evals=None):
        """ Given a set of eval_files instantiate an evaluator object to analyze the evals. """
        self.eval_files = eval_files
        self.eval_dfs = [EvaluationDataframe(datafolder_fp, f, filename_map) for f in eval_files]
        self.max_evals = max_evals
        self.batch_size = batch_size

    def evaluate(self, model, verbose=False, save_verbose=False):
        """ 
        Evaluate the model on the eval files. 
        model: The vLLM model to evaluate.
        """
        result_dicts = []
        verbose_generations = []

        for eval_df in self.eval_dfs:
            print(f"{'='*50}\n Evaluating: {df.filename}\n{'='*50}")
            df = eval_df.df
            results = ResultsDict(eval_df.eval_type)
            for start_idx in enumerate(range(0, len(df), self.batch_size)):
                batch_df = df.iloc[start_idx:start_idx+self.batch_size]
                prompts = [row['prompt'] for _, row in batch_df.iterrows()]
                batch_responses = model.generate(prompts)

                for (index, row), result in zip(batch_df.iterrows(), batch_responses):
                    
                    results.add_result(result.outputs[0].text, row['answer'])
                    
                    if verbose:
                        print(f"{'-'*10}\nPrompt:\n{row['prompt']}\n")
                        print(f"Model Response:\n{result.outputs[0].text}\nGround Truth Answer:\n'{row['answer']}'\n")
                    if save_verbose:
                        verbose_generations.append({
                            "prompt": row['prompt'],
                            "model_response": result.outputs[0].text,
                            "ground_truth": row['answer']
                        })

            result_dicts.append(results.get_final_dict())
            
            print(f"{'-'*50}\nResults for {eval_df.filename}:")
            for key, value in results.get_final_dict().items():
                print(f"{key}: {value}")
            print(f"{'-'*50}\n\n")

            if save_verbose:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(df.datafolder, 'saved_data', f"{df.filename}_{timestamp}.json") 
                with open(save_path, 'w') as f:
                    json.dump(verbose_generations, f, indent=4)

        return result_dicts