import os
import json
import random

import llm_chess.prompts as prompts


class JSONLDataClass():
    def __init__(self, data_dir, filename, task_map, model_version, sys_prompt=None, data_format="chat"):
        """ Load the jsonl file and store useful metadata associated with it. """
        self.filename = filename
        self.trimmed_filename = os.path.splitext(filename)[0]
        self.data_dir = data_dir
        self.sys_prompt = sys_prompt
        self.data_format = data_format
        self.filepath = os.path.join(data_dir, filename)
        self.task_type = next(v for k, v in task_map.items() if filename.startswith(k))
        self.chat_processor = prompts.ChatProcessor(model_version)
        self.data = self._load_data(self.filepath, sys_prompt=sys_prompt)

    def _load_data(self, filepath, sys_prompt=None, shuffle=True):
        """ Load the parquet file and return the dataframe. """
        with open(filepath, 'r') as f:
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]

        data = []
        # Process data and handle multiple possible formats
        if self.data_format == "chat":
            for datum in raw_data:
                prompt, response = self.chat_processor.process_chat(datum['chat'])
                data.append({
                    "prompt": prompt,
                    "response": response,
                    "info": datum['info']
                })
        elif self.data_format == "model_response":
            assert sys_prompt is not None
            for datum in raw_data:
                user_prompt = f"Below is the provided model response:\n\n{datum['model_response']}"
                chat = [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': user_prompt},
                    {'role': 'assistant', 'content': ""}
                ]
                prompt, response = self.chat_processor.process_chat(chat)
                data.append({
                    "prompt": prompt,
                    "response": response,
                    "info": datum['info']
                })

        print(f"Loaded {filepath} with {len(data)} entries.")
        if shuffle:
            random.shuffle(data)

        return data