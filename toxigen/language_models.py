import time
import torch
import subprocess
import requests
import json
from toxigen.alice import beam_search
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
)

class GPT3(object):
    def __init__(self, endpoint_url, api_key):
        self.api_key = api_key
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
                "api-key": self.api_key,
                "Content-Type": "application/json"
            },
            json = payload
        )       
        output = json.loads(r.content)
        return output

    def from_prompt(self, prompt, topk=10, max_tokens=10):
        output = self.__call__(prompt, topk, max_tokens)
        return output["choices"][0]["text"]

class GPT2(object):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained('gpt2', return_dict_in_generate=True)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

    def __call__(self, prompt, stop="\\n", num_responses=10, topk=1):
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        sample_output = self.model.generate(
                input_ids, 
                do_sample=True, 
                max_length=200, 
                top_k=50,
                num_return_sequences=num_responses,
                eos_token_id=int(self.tokenizer(stop)["input_ids"][0]),
                output_scores=True,
        )
        sample_sequences = sample_output.sequences[:, input_ids.shape[-1]:]
        probs = torch.stack(sample_output.scores, dim=1).softmax(-1)
        gen_probs = torch.gather(probs, 2, sample_sequences[:, :, None]).squeeze(-1)
        print(gen_probs)
        assert 2 == 3
        #sample_output = sample_output[0][input_ids.shape[1]:]
        outputs = outputs['choices'][0]['logprobs']['top_logprobs'][0]
        return sample_output
        #return self.tokenizer.decode(sample_output)

class ALICE(object):
    def __init__(self, language_model, classifier, mode, device="cpu"):
        self.classifier = classifier
        self.language_model = language_model
        self.device = device
        self.mode = mode

    def __call__(self, prompt):
        return self.generate(prompt)

    def generate(self, prompt):
        return beam_search(prompt, self.language_model, self.classifier, self.mode, self.device)
