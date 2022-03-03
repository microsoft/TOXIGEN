import torch
import subprocess
import json
from src.alice_decoding import beam_search
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    #AutoModel,
)

class GPT3(object):
    def __init__(self, endpoint_url, api_key):
        self.api_key = api_key
        self.endpoint_url = endpoint_url

    def __call__(self, prompt, stop="<|endoftext|>", num_responses=10, topk=1):
        prompt = prompt.replace("'", "").replace('"', "")
        parameters = {
            "prompt": prompt,
            "max_tokens": 30,
            "temperature": 0.9,
            "n": num_responses,
            "stream": False,
            "logprobs": topk,
            "stop": stop,
        }
        s = f"""curl {self.endpoint_url} -H "Content-Type: application/json" -H "Authorization: Bearer {self.api_key}" -d '{json.dumps(parameters)}'"""
        output = subprocess.check_output(s, shell=True)
        output = json.loads(output)
        outputs = outputs['choices'][0]['logprobs']['top_logprobs'][0]
        return output

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
    def __init__(self, language_model, classifier):
        self.classifier = classifier
        self.language_model = language_model

    def __call__(self, prompt):#, group):
        return self.generate(prompt)#, group)

    def generate(self, prompt):#, group):
        return beam_search(prompt, self.language_model, self.classifier)#, keyword=group)
