import os
import time
import json
import asyncio

from .results_dict import ResultsDict
from .dataclass import JSONLDataClass


class Evaluator():
    def __init__(self, args, task_map, wandb_run):
        """ Given a set of eval_files instantiate an evaluator object to analyze the evals. """
        self.args = args
        self.task_map = task_map
        self.wandb_run = wandb_run
        
        # Load in our various data files
        self.dataclasses = [JSONLDataClass(args.data_dir, filename, task_map, args.llama_version) for filename in args.data_files]

        # Setup various vals just once
        os.makedirs(os.path.join(args.data_dir, 'saved_data'), exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
                

    def evaluate(self, model, verbose=False, save_verbose=True):
        """ 
        Evaluate the model on the eval files. 
        model: The vLLM model client to evaluate.
        """
        result_dicts = []

        # Loop through all our dataclasses and generate / evaluate
        for dataclass in self.dataclasses:
            verbose_generations = []
            
            # Initial setup
            data = dataclass.data
            max_len = len(data) if self.args.max_samples is None else min(len(data), self.args.max_samples)
            print(f"{'='*50}\n Evaluating: {dataclass.trimmed_filename} for {max_len} samples:\n{'='*50}")

            # Set up results dict
            results = ResultsDict(
                task_type = dataclass.task_type,
                filename = dataclass.filename,
                wandb_run = self.wandb_run
            )
            
            # Main eval loop per dataclass
            for start_idx in range(0, max_len, self.args.batch_size):
                data_batch = data[start_idx:min(start_idx+self.args.batch_size, max_len)]
                prompts = [datum['prompt'] for datum in data_batch]
                batch_responses = asyncio.run(model.chat(prompts))

                # Now add results / append response to your dataset
                for idx in range(len(data_batch)):
                    prompt = data_batch[idx]['prompt']
                    response = batch_responses[idx]
                    ground_truth = data_batch[idx]['info']['answer']

                    results.add_result(prompt, response, ground_truth)

                    # Optionally log responses to console for visibility                    
                    if verbose:
                        print(f"{'-'*10}\nPrompt:\n{prompt}\n")
                        print(f"Model Response:\n{response}\nGround Truth Answer:\n'{ground_truth}'\n")
                    if save_verbose:
                        verbose_generations.append({
                            "prompt": prompt,
                            "model_response": response,
                            "ground_truth": ground_truth
                        })

            results, correct_responses = results.get_final_dict(self.args.run_type)
            result_dicts.append(results)
            
            # Finally print results from dataclass evaluation
            print(f"{'-'*50}\nResults for {dataclass.filename}:")
            for key, value in result_dicts[-1].items():
                print(f"{key}: {value}")
            print(f"{'-'*50}\n\n")

            # Save our correct responses if 'rejsampling' task
            if self.args.run_type == 'rejsampling':
                save_path = os.path.join(dataclass.data_dir, 'saved_data', f"{dataclass.trimmed_filename}_correct_{self.timestamp}.json")
                with open(save_path, 'w') as f:
                    json.dump(correct_responses, f, indent=4)
            
            # Also save if save_verbose
            if save_verbose:
                save_path = os.path.join(dataclass.data_dir, 'saved_data', f"{dataclass.trimmed_filename}_all_{self.timestamp}.json")
                with open(save_path, 'w') as f:
                    json.dump(verbose_generations, f, indent=4)

        return result_dicts