import os, time, json, asyncio
from typing import List, Any

from .results_dict import ParserResultsDict
from .dataclass      import JSONFolderDataClass
from .parsing        import coerce_response
from .exceptions     import ParseException


class LLMParser:
    """Parse model generations with an LLM, counting hallucinations etc."""

    # --------------------------------------------------------------------- #
    def __init__(self, args, runtype_mapping, wandb_run):
        self.args          = args
        self.runtype       = args.run_type
        self.wandb_run     = wandb_run
        self.max_retry     = 1
        self.sys_prompt    = runtype_mapping[self.runtype]
        self.dataclasses   = [
            JSONFolderDataClass(args.data_dir, f, args.model_version, self.sys_prompt)
            for f in args.data_files
        ]

        os.makedirs(os.path.join(args.data_dir, "saved_data"), exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

    # --------------------------------------------------------------------- #
    # Public sync wrapper
    def evaluate(self, model, *, verbose: bool = False, save_verbose: bool = True):
        return asyncio.run(
            self._evaluate_async(model, verbose=verbose, save_verbose=save_verbose)
        )

    # --------------------------------------------------------------------- #
    # Core async evaluation
    async def _evaluate_async(self, model, *, verbose: bool, save_verbose: bool):
        results_all: List[dict[str, Any]] = []
        for dc in self.dataclasses:
            print(f"{'='*60}\n Evaluating {dc.trimmed_foldername}\n{'='*60}")
            rd   = ParserResultsDict(self.runtype, dc.trimmed_foldername, self.wandb_run)
            lock = asyncio.Lock()        # protect shared rd / verbose list

            max_samples = (
                len(dc.data)
                if self.args.max_samples is None
                else min(len(dc.data), self.args.max_samples)
            )
            batch_size  = self.args.batch_size
            batches     = range(0, max_samples, batch_size)

            #        prompt | raw | parsed | info
            verbose_store: list[dict[str, Any]] = []

            # ----------------------------------------
            async def _chat(prompts: list[str]) -> list[str]:
                """Always go through the same chat entry point."""
                chat = await model.chat(prompts)
                print(chat)
                return chat

            # ----------------------------------------
            async def _stream_parse(datum, raw_resp, max_retry=1):
                """Parse a single datum, reprompting on ParseException."""
                prompt_txt = datum["prompt"]
                info       = datum["info"]
                attempts   = 0
                cur_raw    = raw_resp
                while True:
                    try:
                        parsed = coerce_response(cur_raw, self.runtype, info=info)
                        async with lock:
                            rd.add_result(parsed)
                            if save_verbose:
                                verbose_store.append(
                                    {
                                        "prompt": prompt_txt,
                                        "model_response": cur_raw,
                                        "parsed_response": parsed,
                                        "info": info,
                                    }
                                )
                        if verbose:  # avoid big lock for prints
                            print(f"{'-'*12}\nPrompt:\n{prompt_txt}\n\n"
                                  f"Raw:\n{cur_raw}\n\nParsed:\n{parsed}\n")
                        return  # success
                    except ParseException as e:
                        async with lock:
                            rd.results["Error: Reprompt"] += 1
                        attempts += 1
                        if attempts > max_retry:
                            return  # give up – counted already
                        cur_raw = (
                            await _chat(
                                [
                                    f"ERROR: {e}\n\nInitial Prompt:\n{prompt_txt}\n\n"
                                    f"Model Response:\n{cur_raw}"
                                ]
                            )
                        )[0]
                    except Exception:        # any other unexpected failure
                        async with lock:
                            rd.results["Error: Other"] += 1
                        return

            # ----------------------------------------
            for start in batches:
                chunk        = dc.data[start : start + batch_size]
                raw_responses = await _chat([d["prompt"] for d in chunk])

                # fire-and-forget parsing tasks
                await asyncio.gather(
                    *(
                        _stream_parse(d, raw_responses[i], max_retry=self.max_retry)
                        for i, d in enumerate(chunk)
                    )
                )

            # -------- per-folder wrap-up --------
            final = rd.get_final_dict()
            results_all.append(final)
            print(f"{'-'*48}\nResults ({dc.trimmed_foldername}):")
            for k, v in final.items():
                print(f"{k}: {v}")
            print(f"{'-'*48}\n")

            if save_verbose:
                path = os.path.join(
                    dc.data_dir,
                    "saved_data",
                    f"{dc.trimmed_foldername}_{self.runtype}_{self.timestamp}.json",
                )
                with open(path, "w") as f:
                    json.dump(verbose_store, f, indent=2)

        return results_all