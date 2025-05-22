from .exceptions import ParseException, IllegalMoveException
from .parsing import extract_solution, coerce_response


class ResultsDict():
    def __init__(self, task_type, filename, wandb_run):
        self.task_type = task_type
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.results = self._instantiate_dict()

    def add_result(self, model_response, ground_truth):
        try:
            self.results["Total Samples"] += 1
            if self.task_type == "choose_from_n":
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                self.results["Correct"] += int(predicted_answer == ground_truth['answer'])
                if predicted_answer != ground_truth['answer']:
                    if predicted_answer in ground_truth['candidates']:
                        self.results["Incorrect"] += 1
                    else:
                        raise IllegalMoveException("Predicted move is not in the provided moves.")
            
            elif self.task_type == 'produce_list':
                self.results["Total Ground Truth Legal Moves"] += len(ground_truth['answer'])
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)

                num_right = 0
                already_guessed = set()
                for move in predicted_answer:
                    if move in ground_truth['answer'] and move not in already_guessed:
                        already_guessed.add(move)
                        num_right += 1
                        self.results["Predicted Ground Truth Legal Moves"] += 1
                    else:
                        self.results["Illegal Moves"] += 1
                
            elif self.task_type == 'predict_singlemove':
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                sorted_answers = sorted(ground_truth['answer'].items(), key=lambda x: x[1])
                
                if predicted_answer in sorted_answers.keys():
                    self.results["Legal Moves Provided"] += 1
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_answers) if move == predicted_answer)
                    self.results["Cumulative Rank of Moves Provided"] += predicted_move_idx/len(sorted_answers)
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

        return self.results

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