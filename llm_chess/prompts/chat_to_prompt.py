import os

# See https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/ for details
LLAMA_3_SPECIAL_TOKENS = {
    "start_header": "<|start_header_id|>",
    "end_header": "<|end_header_id|>",
    "end_of_turn": "<|eot_id|>"
}

# See https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/ for details
LLAMA_4_SPECIAL_TOKENS = {
    "start_header": "<|header_start|>",
    "end_header": "<|header_end|>",
    "end_of_turn": "<|eot|>"
}

# See https://huggingface.co/Qwen/Qwen3-8B?chat_template=default for details
QWEN_3_SPECIAL_TOKENS = {
    "start_header": "<|im_start|>",
    "end_header": "<|im_end|>",
    "end_of_turn": ""
}

# See https://huggingface.co/Qwen/Qwen3-8B?chat_template=default for details
QWEN_25_SPECIAL_TOKENS = {
    "start_header": "<|im_start|>",
    "end_header": "<|im_end|>",
    "end_of_turn": ""
}




class ChatProcessor():
    def __init__(self, model_version):
        self.loaded_prompts = dict()
        self.model_version = model_version
        self._get_special_tokens(model_version)

    def get_prompt(self, filename):
        """ Checks if prompt has already been cached -- if not, loads in prompt. """
        if filename not in self.loaded_prompts:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            abs_path = os.path.join(dir_path, 'samples', filename)
            with open(abs_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            self.loaded_prompts[filename] = prompt
        return self.loaded_prompts[filename]

    def process_chat(self, chat):
        full_prompt = ""    # vLLM automatically prepends bot token 
        response = ""
        
        # Case where we don't want to process anything.
        # Returning just 'chat' because we don't want to require 'chat' to be in a particular dict format
        if self.special_tokens is None:
            return chat
        
        for role, content in chat:
            # Always add the header (even if it is assistant)
            full_prompt += self._add_header(role) 
            if role == 'system':
                if content.endswith('.txt') and all(c not in content for c in r'\/:*?"<>|'):  # Check if valid .txt file
                    full_prompt = full_prompt + self.get_prompt(content) + self.special_tokens['end_of_turn']
                else:
                    full_prompt = full_prompt + content + self.special_tokens['end_of_turn']
            elif role == 'user':
                full_prompt = full_prompt + content + self.special_tokens['end_of_turn']
            elif role == 'assistant':
                response = content
            else:
                raise(ValueError(f"Role must be one of following: system, user, assistant. Currently set as {role}."))

        return full_prompt, response
    
    # ===========================================================
    # Private Helpers
    # ===========================================================
    
    def _get_special_tokens(self, model_version):
        if model_version == "llama3":
            self.special_tokens = LLAMA_3_SPECIAL_TOKENS
        elif model_version == "llama4":
            self.special_tokens = LLAMA_4_SPECIAL_TOKENS
        elif model_version == "qwen3":
            self.special_tokens = QWEN_3_SPECIAL_TOKENS
        elif model_version == "qwen25":
            self.special_tokens = QWEN_25_SPECIAL_TOKENS
        elif model_version is None:
            self.special_tokens = None
        else:
            raise("model_version must be either 'llama3', 'llama4', or 'qwen3'.")

    def _add_header(self, role):
        return self.special_tokens['start_header'] + role + self.special_tokens['end_header']