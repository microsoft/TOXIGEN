import requests
import json
from toxigen.alice import beam_search
class GPT3(object):
    def __init__(self, endpoint_url, apikey):
        self.apikey = apikey
        self.endpoint_url = endpoint_url

    def __call__(self, prompt, topk=1, max_tokens=1):
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt = [p.replace("'", "").replace('"', "") for p in prompt]
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.9,
            "n": 1,
            "stream": False,
            "logprobs": topk,
            "stop": ["<|endoftext|>", "\\n"]
        }
        r = requests.post(self.endpoint_url,
            headers={
                "Authorization": f"Bearer {self.apikey}",
                "Content-Type": "application/json"
            },
            json = payload
        )       
        output = json.loads(r.content)
        return output

    def from_prompt(self, prompt, topk=10, max_tokens=10):
        output = self.__call__(prompt, topk, max_tokens)
        return output["choices"][0]["text"]

class ALICE(object):
    def __init__(self, language_model, classifier, mode, device="cpu"):
        self.classifier = classifier
        self.language_model = language_model
        self.device = device
        self.mode = mode

    def __call__(self, prompt):
        return self.generate(prompt)

    def generate(self, prompt):
        if self.mode == "neutral":
            flag = 0
        else:
            flag = 1
        return beam_search(prompt, self.language_model, self.classifier, flag, self.device)
