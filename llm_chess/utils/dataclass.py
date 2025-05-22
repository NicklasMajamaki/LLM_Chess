import os
import json
import random

import llm_chess.prompts as prompts


class JSONLDataClass():
    def __init__(self, data_dir, filename, task_map, llama_version):
        """ Load the jsonl file and store useful metadata associated with it. """
        self.filename = filename
        self.trimmed_filename = os.path.splitext(filename)[0]
        self.data_dir = data_dir
        self.filepath = os.path.join(data_dir, filename)
        self.task_type = next(v for k, v in task_map.items() if filename.startswith(k))
        self.chat_processor = prompts.ChatProcessor(llama_version)
        self.data = self._load_data(self.filepath)

    def _load_data(self, filepath, shuffle=True):
        """ Load the parquet file and return the dataframe. """
        with open(filepath, 'r') as f:
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]

        # Process data
        data = []
        for datum in raw_data:
            prompt, response = self.chat_processor.process_chat(datum['chat'])
            data.append({
                "prompt": prompt,
                "response": response,
                "info": datum['info']
            })
        print(f"Loaded {filepath} with {len(data)} entries.")

        # Optionally shuffle data
        if shuffle:
            random.shuffle(data)

        return data