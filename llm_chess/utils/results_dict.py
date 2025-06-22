from .exceptions import ParseException, IllegalMoveException
from .parsing import extract_solution, coerce_response


class ResultsDict():
    def __init__(self, task_type, filename, wandb_run):
        self.task_type = task_type
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.results = self._instantiate_dict()
        self.correct_responses = []

    def add_result(self, prompt, model_response, ground_truth):
        try:
            self.results["Total Samples"] += 1
            if self.task_type == "choose_from_n":
                answer = ground_truth['answer']
                candidates = ground_truth['candidates']
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                
                # Determine correctness
                if predicted_answer == answer:
                    self.results["Correct"] += 1
                    self.correct_responses.append({
                        "prompt": prompt,
                        "completion": model_response,
                        "ground_truth": ground_truth
                    })
                else:
                    if predicted_answer in candidates:
                        self.results["Incorrect"] += 1
                    else:
                        raise IllegalMoveException("Predicted move is not in the provided moves.")
            
            elif self.task_type == 'produce_list':
                answer = ground_truth
                self.results["Total Ground Truth Legal Moves"] += len(answer)
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)

                # Compute correctness
                num_right = 0
                already_guessed = set()
                for move in predicted_answer:
                    if move in answer and move not in already_guessed:
                        already_guessed.add(move)
                        num_right += 1
                        self.results["Predicted Ground Truth Legal Moves"] += 1
                    else:
                        self.results["Illegal Moves"] += 1

                # Only append to correct_response if all correct
                if already_guessed == set(answer) and len(predicted_answer) == len(answer):
                    self.correct_responses.append({
                        "prompt": prompt,
                        "completion": model_response,
                        "ground_truth": ground_truth
                    })
                
            elif self.task_type == 'predict_singlemove':
                answer = ground_truth

                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                sorted_answers = sorted(answer.items(), key=lambda x: x[1])
                
                if predicted_answer in answer:
                    self.results["Legal Moves Provided"] += 1
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_answers) if move == predicted_answer)
                    self.results["Cumulative Rank of Moves Provided"] += predicted_move_idx/len(sorted_answers)

                    # Only keep if > 0.7 (w/in top 30% of moves)
                    if predicted_move_idx/len(sorted_answers) > 0.7:
                        self.correct_responses.append({
                            "prompt": prompt,
                            "completion": model_response,
                            "ground_truth": ground_truth
                        })
                else:
                    raise IllegalMoveException("Predicted move is not in the legal moves.")
        
        # Exception handling to log various errors     
        except Exception as e:
            if isinstance(e, ParseException):
                self.results["Error: Parsing"] += 1
            elif isinstance(e, IllegalMoveException):
                self.results["Error: Illegal Move"] += 1
            else:
                self.results["Error: Other"] += 1
        
    def get_final_dict(self, run_type):
        """ run_type is either 'eval' or 'rejsampling' -- used for wandb logging. """
        run_type = run_type.capitalize()

        if self.task_type == "choose_from_n":
            total = self.results["Total Samples"]
            self.results["Accuracy"] = self._safe_div(self.results["Correct"], total)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Accuracy": self.results["Accuracy"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"],
                })

        elif self.task_type == "produce_list":
            gt_total = self.results["Total Ground Truth Legal Moves"]
            illegal = self.results["Illegal Moves"]
            total = self.results["Total Samples"]
            self.results["Percent Legal Moves Predicted"] = self._safe_div(self.results["Predicted Ground Truth Legal Moves"], gt_total)
            self.results["Ratio of Legal to Illegal Moves"] = self._safe_div(self.results["Predicted Ground Truth Legal Moves"], illegal)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Percent Legal Moves Predicted": self.results["Percent Legal Moves Predicted"],
                    f"{run_type} - {self.trimmed_filename}/Ratio of Legal to Illegal Moves": self.results["Ratio of Legal to Illegal Moves"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        elif self.task_type == "predict_singlemove":
            legal = self.results["Legal Moves Provided"]
            total = self.results["Total Samples"]
            self.results["Avg. Rank of Move Provided"] = self._safe_div(self.results["Cumulative Rank of Moves Provided"], legal)
            self.results["Percent Legal Moves Provided"] = self._safe_div(legal, total)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Avg. Rank of Move Provided": self.results["Avg. Rank of Move Provided"],
                    f"{run_type} - {self.trimmed_filename}/Percent Legal Moves Provided": self.results["Percent Legal Moves Provided"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        return self.results, self.correct_responses

    # =================================================
    # Internal helper functions
    # =================================================
    def _instantiate_dict(self):
        if self.task_type == "choose_from_n":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Correct": 0,
                "Incorrect": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "produce_list":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Total Ground Truth Legal Moves": 0,
                "Predicted Ground Truth Legal Moves": 0,
                "Illegal Moves": 0,
                "Error: Parsing": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "predict_singlemove":
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
            raise ValueError(f"Undefined task type: {self.task_type}")

    def _safe_div(self, x, y, default=0): 
        return x / y if y else default
    


# =============================================
# Results Dict for LLM Parsing Cases
# =============================================
class ParserResultsDict():
    def __init__(self, task_type, filename, wandb_run):
        self.task_type = task_type
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.results = self._instantiate_dict()

    def add_result(self, parsed_response):
        self.results["Total Responses Parsed"] += 1
        if self.task_type == "hallucination":
            for k, v in parsed_response.items():
                self.results[k] += v

        elif self.task_type == "reasoning_strategy":
            for k, v in parsed_response.items():
                self.results[f"Count: {k}"] += v

    def get_final_dict(self):
        """ Return finalized dict and log to wandb. """
        if self.task_type == "hallucination":
            hallucination_percent = self._safe_div(self.results['Count: Hallucinations'],  self.results['Total Moves'])
            average_moves_per_response = self._safe_div(self.results['Total Moves'], self.results['Total Responses Parsed'])
            parsing_moves_error_rate = self._safe_div(self.results['Error: Parsing Move'], self.results['Total Moves']) 
            percent_reprompts = self._safe_div(self.results['Error: Reprompt'], self.results['Total Moves'])

            self.results['Hallucination Percent'] = hallucination_percent
            self.results['Ave. Moves Per Response'] = average_moves_per_response
            self.results['Parsing Moves Error Rate'] = parsing_moves_error_rate
            self.results['Percent Reprompts'] = percent_reprompts
            
            if self.wandb_run:
                self.wandb_run.log({
                    f"Hallucination / Hallucination Percent": self.results["Hallucination Percent"],
                    f"Hallucination / Ave. Moves Per Response": self.results["Ave. Moves Per Response"],
                    f"Hallucination / Parsing Moves Error Rate": self.results["Parsing Moves Error Rate"],
                    f"Hallucination / Percent Reprompts": self.results["Percent Reprompts"]                
                })
        
        elif self.task_type == "reasoning_strategy":
            self.results['Percent Enumeration'] = self._safe_div(self.results['Count: Enumeration'], self.results['Total Responses Parsed'])
            self.results['Percent Tree Search'] = self._safe_div(self.results['Count: Tree Search'], self.results['Total Responses Parsed'])
            self.results['Percent Backtracking'] = self._safe_div(self.results['Count: Backtracking'], self.results['Total Responses Parsed'])
            self.results['Percent Self Correction'] = self._safe_div(self.results['Count: Self Correction'], self.results['Total Responses Parsed'])
            self.results['Percent Subgoal Setting'] = self._safe_div(self.results['Count: Subgoal Setting'], self.results['Total Responses Parsed'])
            self.results['Percent Verification'] = self._safe_div(self.results['Count: Verification'], self.results['Total Responses Parsed'])
            self.results['Percent Reprompts'] = self._safe_div(self.results['Error: Reprompt'], self.results['Total Responses Parsed'])
            
            if self.wandb_run:
                self.wandb_run.log({
                    f"Reasoning Strategy / Percent Enumeration": self.results["Percent Enumeration"],
                    f"Reasoning Strategy / Percent Tree Search": self.results["Percent Tree Search"],
                    f"Reasoning Strategy / Percent Backtracking": self.results["Percent Backtracking"],
                    f"Reasoning Strategy / Percent Self Correction": self.results["Percent Self Correction"],
                    f"Reasoning Strategy / Percent Subgoal Setting": self.results["Percent Subgoal Setting"],
                    f"Reasoning Strategy / Percent Verification": self.results["Percent Verification"],
                    f"Reasoning Strategy / Percent Reprompts": self.results["Percent Reprompts"],
                })

        return self.results


    # =================================================
    # Internal helper functions
    # =================================================
    def _instantiate_dict(self):
        if self.task_type == "hallucination":
            return {
                "Filename": self.filename,
                "Total Responses Parsed": 0,
                "Total Moves": 0,
                "Count: Correct Moves": 0,
                "Count: Hallucinations": 0,
                "Error: Reprompt": 0,
                "Error: Parsing Move": 0,
                "Error: Other": 0
            }
        elif self.task_type == "reasoning_strategy":
            return {
                "Filename": self.filename,
                "Total Responses Parsed": 0,
                "Count: Enumeration": 0,
                "Count: Tree Search": 0,
                "Count: Backtracking": 0,
                "Count: Self Correction": 0,
                "Count: Subgoal Setting": 0,
                "Count: Verification": 0,
                "Error: Reprompt": 0,
                "Error: Other": 0,
            }
        else:
            raise ValueError(f"Undefined task type: {self.task_type}")

    def _safe_div(self, x, y, default=0): 
        return x / y if y else default
    