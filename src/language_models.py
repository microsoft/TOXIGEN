import torch
import subprocess
import json
from src.alice_decoding import beam_search
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer
)
from transformers import pipeline

class GPT3(object):
    def __init__(self, apikey):
        self.apikey = apikey

    def __call__(self, prompt, stop="<|endoftext|>", num_responses=10, topk=1):
        endpoint_url = "https://api.openai.com/v1/engines/davinci-msft/completions"
        #endpoint_url = "https://gpt3-babel.eastus.inference.ml.azure.com/v1/engines/davinci/completions"
        prompt = prompt.replace("'", "").replace('"', "")
        #prompt = [p.replace('"', "").replace("'", "") for p in prompt]
        parameters = {
            "prompt": prompt,
            "max_tokens": 30,
            "temperature": 0.9,
            "n": num_responses,
            "stream": False,
            "logprobs": topk,
            "stop": stop,
        }
        s = f"""curl {endpoint_url} -H "Content-Type: application/json" -H "Authorization: Bearer {apikey}" -d '{json.dumps(parameters)}'"""
        output = subprocess.check_output(s, shell=True)
        output = json.loads(output)
        return output

class GPT2(object):
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.eos_token = "\n"

    def __call__(self, prompt, stop="\n", num_responses=10, topk=1):
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        sample_output = self.model.generate(
                input_ids, 
                do_sample=True, 
                max_length=200, 
                top_k=50,
        )
        sample_output = sample_output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(sample_output)

class ALICE(object):
    def __init__(self, language_model, classifier):#, apikey):
        self.classifier = classifier
        self.language_model = language_model

    def __call__(self, prompt):
        return self.generate(prompt)

    def generate(self, prompt, group, device):
        sentence = beam_search(language_model, prompt, keyword=group, classifier_dir=self.classifier, device=device)
        return sentence

