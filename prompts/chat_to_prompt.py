import os

# See https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/ for details
LLAMA_3_SPECIAL_TOKENS = {
    "begin_of_text": "<|begin_of_text|>",
    "end_of_text": "<|end_of_text|>",
    "start_header": "<|start_header_id|>",
    "end_header": "<|end_header_id|>",
    "end_of_turn": "<|eot_id|>"
}

# See https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/ for details
LLAMA_4_SPECIAL_TOKENS = {
    "begin_of_text": "<|begin_of_text|>",
    "end_of_text": "<|end_of_text|>",
    "start_header": "<|header_start|>",
    "end_header": "<|header_end|>",
    "end_of_turn": "<|eot|>"
}


class ChatProcessor():
    def __init__(self, llama_version):
        self.loaded_prompts = dict()
        self.llama_version = llama_version
        self._get_special_tokens(llama_version)
    
    def _get_special_tokens(self, llama_version):
        if llama_version == "llama3":
            self.special_tokens = LLAMA_3_SPECIAL_TOKENS
        elif llama_version == "llama4":
            self.special_tokens = LLAMA_4_SPECIAL_TOKENS
        else:
            raise("llama_version must be either 'llama3' or 'llama4'.")

    def _add_header(self, role):
        return self.special_tokens['start_header'] + role + self.special_tokens['end_header']

    def _get_prompt(self, filename):
        """ Checks if prompt has already been cached -- if not, loads in prompt. """
        if filename not in self.loaded_prompts:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            abs_path = os.path.join(dir_path, filename)
            with open(abs_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            self.loaded_prompts[filename] = prompt
        return self.loaded_prompts[filename]

    def process_chat(self, chat):
        full_prompt = self.special_tokens['begin_of_text']
        response = ""
        for role, content in chat:
            # Always add the header (even if it is assistant)
            full_prompt += self._add_header(role) 
            if role == 'system':
                if content.endswith('.txt') and all(c not in content for c in r'\/:*?"<>|'):  # Check if valid .txt file
                    full_prompt = full_prompt + self._get_prompt(content) + self.special_tokens['end_of_turn']
                else:
                    full_prompt = full_prompt + content + self.special_tokens['end_of_turn']
            elif role == 'user':
                full_prompt += content
            elif role == 'assistant':
                response = content
            else:
                raise(ValueError(f"Role must be one of following: system, user, assistant. Currently set as {role}."))

        return full_prompt, response