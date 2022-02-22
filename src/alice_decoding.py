# Beam search modified from HuggingFace implementation
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import subprocess
import itertools
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import stopwords

def omit_(token, score, list1,list2):
    if token not in list1 and token.replace(' ','') not in list1 and token in list2:
       return -100
    return score

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

def beam_search(prompt,
                end_token='<|endoftext|>',
                weights=[.5,.5],
                keyword='latino',
                use_class=True,
                num_beams=10,
                vocab_size=100,
                max_length=5,
                temperature=.9,
                length_penalty=1,
                classifier_dir="./HateBERT_offenseval",
                device="cuda"):
    """Generate sequences for each example with beam search."""
    pad_token_id = '<|pad|>'
    eos_token_ids = [end_token]
    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(1)
    ]
    stops = list(stopwords.words('english'))
    stops += [keyword, ' \\','\\','n','.',',','!']
    # scores for each sentence in the beam
    beam_scores = torch.zeros((1, num_beams), dtype=torch.float, device='cpu')
    # Greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = None

    # done sentences
    done = [False for _ in range(1)]

    step = 1
    start_index = len(prompt.split(' '))
    input_ids = [prompt] * num_beams
    outputs = {}
    while 'choices' not in outputs.keys():
           outputs = language_model([prompt], stop=end_token, num_responses=1, topk=num_beams)
           #outputs = query_gpt3([prompt],api_key, endpoint_url, end_token, num_responses=1,topk=num_beams)
    outputs = outputs['choices'][0]['logprobs']['top_logprobs'][0]
    tokens = list(outputs.keys())
    tokens = [(k,outputs[k]) for k in tokens]
    for i in range(len(tokens)):
        beam_scores[i] = tokens[i][1]
        input_ids[i] += tokens[i][0]
    if use_class:
       tokenizer = AutoTokenizer.from_pretrained(classifier_dir)
       hatebert = AutoModelForSequenceClassification.from_pretrained(classifier_dir)
       hatebert.eval()
       hatebert.to(device)
    while step < max_length:
        outputs = {}
        while 'choices' not in outputs.keys():
              try:
                  outputs = query_gpt3(input_ids, api_key, endpoint_url, end_token, num_responses=1,topk=vocab_size)
              except:
                  continue
        scores = [outputs['choices'][i]['logprobs']['top_logprobs'] for i in range(num_beams)]
        full_names = [[list(x.keys()) for x in scores[i]] for i in range(num_beams)]
        scores = [[list(x.values()) for x in scores[i]] for i in range(num_beams)]
        scores_ = torch.Tensor([[[omit_(full_names[i][0][j], scores[i][0][j], stops, prompt) for j in range(len(scores[i][0]))]] for i in range(num_beams)])
        scores = scores_.view(num_beams * 1, vocab_size)
        full_names = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(full_names))))
        # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(1, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        next_tokens_names = [full_names[int(next_tokens[0][i])] for i in range(len(next_tokens[0]))]
        assert next_scores.size()[-1] == len(next_tokens_names) == 2 * num_beams
        if use_class == True:
           bert_inputs = [tokenizer.encode(' '.join(input_ids[t // vocab_size ].split(' ')[start_index:]) + full_names[t]) for t in next_tokens[0]]
           pad_len = max([len(t) for t in bert_inputs])
           bert_inputs = torch.LongTensor([b + [0] * (pad_len - len(b)) for b in bert_inputs])
           logits = torch.nn.functional.log_softmax(hatebert(bert_inputs.to(device)).logits)[:,1].cpu()
           next_scores = (next_scores * weights[0]) + (logits * weights[1])

        # next batch beam content
        # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []
        # for each sentence
        for batch_idx in range(1):

            # if we are done with this sentence
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item()
            )
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_ids is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            check_ = [next_tokens[batch_idx][i] // vocab_size for i in range(len(next_tokens[batch_idx]))]
            for idx, score in zip(next_tokens[batch_idx], next_scores[batch_idx]):
                # get beam and word IDs
                beam_id = idx // vocab_size
                token_id = full_names[idx]
                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if eos_token_ids is not None and token_id in eos_token_ids:
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id], score.item(),
                    )
                else:
                    # add next predicted word if it is not eos_token
                    next_sent_beam.append((score, token_id, effective_beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == 1 * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = [x[1] for x in next_batch_beam]
        beam_idx = [x[2] for x in next_batch_beam]
        # re-order batch
        input_ids = [input_ids[i] for i in beam_idx]
        input_ids = [input_ids[i] + beam_tokens[i] for i in range(len(input_ids))]

        # re-order internal states
        if past:
            past = self._reorder_cache(past, beam_idx)

        # stop when we are done with each sentence
        if all(done):
            break

        # update current length
        step = step + 1

    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(1):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        #if eos_token_ids is not None and all(
        #    (token_id % vocab_size).item() not in eos_token_ids for token_id in next_tokens[batch_idx]
        #):
            #import pdb; pdb.set_trace()
            #assert torch.all(
            #    next_scores[batch_idx, :num_beams] == beam_scores.view(1, num_beams)[batch_idx]
            #), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
            #    next_scores[:, :num_beams][batch_idx], beam_scores.view(1, num_beams)[batch_idx]
            #)
        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)
    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch

    best_all = []
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        best_all.append(sorted(hypotheses.beams, key=lambda x: x[0],reverse=True))
    return [p[-1] for p in best_all[0]]

def query_gpt3(prompt, apikey, endpoint_url="https://gpt3-babel.eastus.inference.ml.azure.com/v1/engines/davinci/completions", end_token='<|endoftext|>',num_responses=1, topk=0):
    prompt = [p.replace('"', "").replace("'", "") for p in prompt]
    parameters = {
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.9,
        "n": num_responses,
        "stream": False,
        "logprobs": topk,
        "stop": end_token,
    }
    s = f"""curl {endpoint_url} -H "Content-Type: application/json" -H "Authorization: Bearer {apikey}" -d '{json.dumps(parameters)}'"""
    output = subprocess.check_output(s, shell=True)
    output = json.loads(output)
    return output
