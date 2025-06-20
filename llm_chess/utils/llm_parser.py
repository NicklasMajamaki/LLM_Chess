import os
import time
import json
import asyncio
from typing import List, Any

from .results_dict import ParserResultsDict
from .dataclass import JSONLFolderDataClass
from .parsing import coerce_response
from .exceptions import ParseException




class LLMParser():
    """ Wrapper that structures using an LLM to parse previous model generations to extract more nuanced information (e.g., # hallucinations, reasoning strategies used). """

    def __init__(self, args, runtype_mapping, wandb_run):
        """ Given a set of eval_files instantiate an evaluator object to analyze the evals. """
        self.args = args
        self.runtype_mapping = runtype_mapping
        self.wandb_run = wandb_run
        
        # Load in our various data files
        self.dataclasses = [JSONLFolderDataClass(args.data_dir, foldername, args.model_version, sys_prompt=runtype_mapping[args.run_type], data_format="model_response") for foldername in args.data_files]
        # Data will be in format "prompt, response, info" for the keys

        # Setup various vals just once
        os.makedirs(os.path.join(args.data_dir, 'saved_data'), exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

    def evaluate(self, model, verbose: bool=False, save_verbose: bool=True):
        """Sync wrapper so the external API remains unchanged."""
        return asyncio.run(
            self._evaluate_async(model, verbose=verbose, save_verbose=save_verbose)
        )

    # =========================================================
    # Async Helper
    # =========================================================
    async def _evaluate_async(
        self, model, verbose: bool=False, save_verbose: bool=True
    ) -> List[dict[str, Any]]:
        """True async implementation - can be awaited or called via evaluate()."""

        # --------------------------------------------------------------------
        # Locks to protect shared mutable state
        # --------------------------------------------------------------------
        results_lock = asyncio.Lock()
        verbose_lock = asyncio.Lock()

        # --------------------------- Internal Helpers ---------------------------
        async def _ask_llm(prompt: str) -> str:
            """Send a single prompt; unwrap the list that vLLM returns."""
            return (await model.chat([prompt]))[0]

        async def _parse_with_retry(prompt: str, run_type: str, max_retry: int = 1):
            raw_response = await _ask_llm(prompt)
            attempts = 1
            while True:
                try:
                    return coerce_response(raw_response, run_type), raw_response, attempts
                except Exception as e:
                    if isinstance(e, ParseException):
                        async with results_lock:
                            results['Error: Reprompt'] += 1
                        if attempts > max_retry:
                            raise
                        reprompt = (
                            f"ERROR: {e}\n\nInitial Prompt:\n{prompt}\n\nModel Response:\n{raw_response}"
                        )
                        raw_response = await _ask_llm(reprompt)
                        attempts += 1
                    else:
                        async with results_lock:
                            results['Error: Other'] += 1

        # --------------------------------------------------------------------
        results_dicts = []
        for dataclass in self.dataclasses:
            verbose_generations = []
            data      = dataclass.data
            max_len   = len(data) if self.args.max_samples is None else min(len(data), self.args.max_samples)
            batch_sz  = self.args.batch_size
            run_type  = self.args.run_type

            print(f"{'='*50}\n Evaluating: {dataclass.trimmed_foldername} "
                  f"for {max_len} samples\n{'='*50}")

            results = ParserResultsDict(
                task_type = run_type,
                filename  = dataclass.trimmed_foldername,
                wandb_run = self.wandb_run
            )

            # -------- MAIN LOOP (batched async) --------
            for start in range(0, max_len, batch_sz):
                chunk     = data[start : start + batch_sz]
                prompts   = [d["prompt"] for d in chunk]

                batch_raw = await model.chat(prompts)
                tasks = []

                for i, datum in enumerate(chunk):
                    raw_resp = batch_raw[i]

                    async def _handle(idx=i, d=datum, raw=raw_resp):
                        prompt_txt  = d["prompt"]
                        info        = d["info"]

                        try:
                            parsed = coerce_response(raw, run_type, info=info)
                        except Exception as e:
                            if isinstance(e, ParseException):
                                async with results_lock:
                                    results['Error: Reprompt'] += 1
                                parsed, raw_fixed, _ = await _parse_with_retry(prompt_txt, run_type)
                                raw = raw_fixed
                            else:
                                async with results_lock:
                                    results['Error: Other'] += 1
                                return

                        async with results_lock:
                            results.add_result(parsed)

                        if save_verbose:
                            async with verbose_lock:
                                verbose_generations.append(
                                    {
                                        "prompt": prompt_txt,
                                        "model_response": raw,
                                        "parsed_response": parsed,
                                        "info": info,
                                    }
                                )

                        if verbose:
                            print(f"{'-'*10}\nOriginal Prompt:\n{prompt_txt}\n"
                                  f"Model Response:\n{raw}\n\nParsed Response:\n{parsed}\n")

                    tasks.append(asyncio.create_task(_handle()))

                await asyncio.gather(*tasks)

            # ------- wrap-up per dataclass -------
            results = results.get_final_dict(run_type)
            results_dicts.append(results)

            print(f"{'-'*50}\nResults for {dataclass.trimmed_foldername}:")
            for k, v in results.items():
                print(f"{k}: {v}")
            print(f"{'-'*50}\n")

            if save_verbose:
                path = os.path.join(
                    dataclass.data_dir,
                    "saved_data",
                    f"{dataclass.trimmed_foldername}_{self.args.run_type}_{self.timestamp}.json",
                )
                with open(path, "w") as f:
                    json.dump(verbose_generations, f, indent=4)

        return results_dicts