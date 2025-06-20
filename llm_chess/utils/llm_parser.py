import os
import time
import json
import asyncio

from .results_dict import ParserResultsDict
from .dataclass import JSONLDataClass
from .parsing import coerce_response
import llm_chess.prompts as prompts


RUNTYPE_SYSPROMPT_MAPPING = {
    'hallucination': 'hallucinations_sysprompt.txt', 
    'reasoning_strategy': 'reasoning_strategies_sysprompt.txt'
}


class LLMParser():
    """ Wrapper that structures using an LLM to parse previous model generations to extract more nuanced information (e.g., # hallucinations, reasoning strategies used). """

    def __init__(self, args, task_map, wandb_run):
        """ Given a set of eval_files instantiate an evaluator object to analyze the evals. """
        self.args = args
        self.task_map = task_map
        self.wandb_run = wandb_run
        
        # Load in our various data files
        self.dataclasses = [JSONLDataClass(args.data_dir, filename, task_map, args.model_version, sys_prompt=RUNTYPE_SYSPROMPT_MAPPING[args.run_type], data_format="model_response") for filename in args.data_files]
        # Data will be in format "prompt, response, info" for the keys

        # Setup various vals just once
        os.makedirs(os.path.join(args.data_dir, 'saved_data'), exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        
    def evaluate(self, model, verbose=False, save_verbose=True):
        """ Run through our data and parse / evaluate parsed values using an LLM. """
        results_dicts = []
        
        # Loop through all our dataclasses and generate / evaluate
        for dataclass in self.dataclasses:
            verbose_generations = []
            
            # Initial setup
            data = dataclass.data
            max_len = len(data) if self.args.max_samples is None else min(len(data), self.args.max_samples)
            print(f"{'='*50}\n Evaluating: {dataclass.trimmed_filename} for {max_len} samples:\n{'='*50}")

            # Set up results dicts
            results = ParserResultsDict(
                task_type = self.args.run_type,
                filename = dataclass.filename,
                wandb_run = self.wandb_run
            )

            # Need to figure out way to handle this such that we can send requests to the model endpoint in batches and handle each returned response on its own. So if there is an error when trying to parse / handle an element, we should be able to catch this error and reprompt the model using it
            # We also need to specify a reprompt depth (i.e., max 1 reprompt)

            # Main eval loop per dataclass
            for start_idx in range(0, max_len, self.args.batch_size):
                data_batch = data[start_idx:min(start_idx+self.args.batch_size, max_len)]
                prompts = [datum['prompt'] for datum in data_batch]
                # This is a standard model chat api endpoint handled by VLLM
                batch_responses = asyncio.run(model.chat(prompts))

                for idx in range(len(data_batch)):
                    original_prompt = data_batch[idx]['prompt']
                    response = batch_responses[idx]
                    prompt_info = data_batch[idx]['info']

                    # Need to figure out when we want to do our parsing
                    # Call 'parse_data' --> This will raise a parsing error if we need to reprompt (should only reprompt up to 1 time)
                    # If it is anything other than a parsing error you should send that error into 'add_result' as we'll incremenet 'error:other' then
                    parsed_response = coerce_response(response, self.args.run_type)
                    results.add_result(parsed_response)

                    # Optionally log responses to console for visibility                    
                    if verbose:
                        print(f"{'-'*10}\nOriginal Prompt:\n{original_prompt}\n")
                        print(f"Model Response:\n{response}\n\nParsed Response:\n'{parsed_response}'\n")
                    if save_verbose:
                        verbose_generations.append({
                            "prompt": original_prompt,
                            "model_response": response,
                            "parsed_response": parsed_response,
                            "info": prompt_info
                        })

            results, correct_responses = results.get_final_dict(self.args.run_type)
            results_dicts.append(results)
            
            # Finally print results from dataclass evaluation
            print(f"{'-'*50}\nResults for {dataclass.filename}:")
            for key, value in results_dicts[-1].items():
                print(f"{key}: {value}")
            print(f"{'-'*50}\n\n")
            
            # Also save if save_verbose
            if save_verbose:
                save_path = os.path.join(dataclass.data_dir, 'saved_data', f"{dataclass.trimmed_filename}_all_{self.timestamp}.json")
                with open(save_path, 'w') as f:
                    json.dump(verbose_generations, f, indent=4)

        return results_dicts


            