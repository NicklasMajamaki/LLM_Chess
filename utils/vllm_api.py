import openai
import asyncio
from typing import List


class vLLMClient:
    def __init__(self, model: str, base_url: str, generation_args: dict, api_key: str = 'sk-no-key-needed'):
        """
        Create a vLLM client instance that we'll call on to chat with the underlying model.

        base_url: url to the vLLM server. Can replace with external endpoint if we want to change our endpoint (e.g., using oai)
        api_key: API key for authentication. Not needed for local vLLM server.
        model: The model we're using for this client instance
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        self.generation_args = generation_args
        self.max_tokens = generation_args.get("max_tokens", 2048)
        self.temperature = generation_args.get("temperature", 0.7)
        self.top_p = generation_args.get("top_p", 0.9)
        self.min_p = generation_args.get("min_p", 0.0)
        self.top_k = generation_args.get("top_k", 40)
        self.repetition_penalty = generation_args.get("repetition_penalty", 1.1)

        self.client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def _chat_single(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty
        )
        return response.choices[0].message.content.strip()

    async def chat(self, prompts: List[str]) -> List[str]:
        """
        Pass in a list of prompts -- this will process multiple chats in parallel.
        """
        tasks = [self._chat_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)