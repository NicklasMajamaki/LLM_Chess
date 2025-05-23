import requests
import asyncio
from typing import List

class vLLMClient:
    def __init__(self, model: str, base_url: str, generation_args: dict, api_key: str = 'sk-no-key-needed'):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        self.generation_args = generation_args
        self.max_tokens = generation_args["max_tokens"]
        self.temperature = generation_args["temperature"]
        self.top_p = generation_args["top_p"]
        self.min_p = generation_args["min_p"]
        self.top_k = generation_args["top_k"]
        self.repetition_penalty = generation_args["repetition_penalty"]

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _chat_single(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty
        }

        response = requests.post(f"{self.base_url}", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()['choices'][0]['text'].strip()

    async def chat(self, prompts: List[str]) -> List[str]:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self._chat_single, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)