import torch
import subprocess
import json
from alice_decoding import beam_search

class GPT3(object):
    def __init__(self, apikey):
        self.apikey = apikey

    def __call__(self, prompt):
        return self.query_gpt3(prompt, self.apikey)

    def query_gpt3(self, prompt, apikey, num_responses=10, topk=1):
        endpoint_url = "https://api.openai.com/v1/engines/davinci-msft/completions"
        prompt = [p.replace('"', "").replace("'", "") for p in prompt]
        parameters = {
            "prompt": prompt,
            "max_tokens": 30,
            "temperature": 0.9,
            "n": num_responses,
            "stream": False,
            "logprobs": topk,
            "stop": "<|endoftext|>",
        }
        s = f"""curl {endpoint_url} -H "Content-Type: application/json" -H "Authorization: Bearer {apikey}" -d '{json.dumps(parameters)}'"""
        output = subprocess.check_output(s, shell=True)
        output = json.loads(output)
        return output

class GPT2(object):
    def __init__(self):
        self.model = 3
        self.tokenizer = 2

    def __call__(self, prompt):
        pass

class ALICE(object):
    def __init__(self, classifier, apikey):
        self.apikey = apikey
        self.classifier = classifier

    def __call__(self, prompt):
        return self.generate(prompt)

    def generate(self, prompt, group, device):
        """
        This is where the complicated beam search stuff goes!
        """
        endpoint_url = "https://api.openai.com/v1/engines/davinci-msft/completions"
        sentence = beam_search(prompt, self.apikey, endpoint_url, keyword=group, classifier_dir=self.classifier, device=device)
        return sentence

