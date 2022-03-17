# ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection
This is the research release for [ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection](https://arxiv.org/abs/2030.12345).
It includes two components: a large-scale dataset containing implicitly toxic and benign sentences mentioning 13 minority groups, and a tool to stress test a given off-the-shelf toxicity classifier. The dataset is generated using a large language model (GPT3). It is intended to be used for training classifiers that learn to detect subtle toxicity that includes no slurs or profanity. The data, methods and trained checkpoint released with this work are intended to be used for research purposes only. 

This repository includes two methods for generating new sentences given a large scale pretrained language model (e.g., GPT3) and an off the shelf classifier:
- **Demonstration-Based Prompting**, where a language model is used to create more data given human provided prompts across different minority groups. 
- **ALICE**, which creates an adversarial set up between a given toxicity classifier and a generator (pretrained language model) to create challenging examples for the classifier and improve its performance. 

**WARNING: This repository contains and discusses content that is offensive or upsetting. All materials are intended to support research that improves toxicity detection methods. Included examples of toxicity do not represent how the authors or sponsors feel about any identity groups.**

## Download the data

[Download the ToxiGen dataset](www.google.com). The full training dataset, including metadata about the prompts.

## Generate data with ToxiGen demonstration-based prompts

To pass the collected prompts in the dataset into the pretrained language model (GPT-3) and generate new sentences, run this command:

```
python generate.py --input_prompt_file <path_to_prompt_file.txt> --language_model GPT3 --output_file <path_to_output_file.txt> --num_generations_per_prompt 10 --openai_api_key <your_api_key>
```

You can choose from a list of [prompt files](./prompts/) that we use in the paper or write your own and point to the file (shown below). A prompt file is a text file with one line per prompt (a string).

## Generate data using ALICE

To generate data using ALICE, it is necessary to choose a generator (GPT3 in our case) and a pre-trained toxicity classifier. We provide a few here and some guidance on adding new classifiers. To generate with ALICE, run this command:

```
python generate.py --input_prompts <path_to_prompt_file.txt> --language_model GPT3 --ALICE True --classifier HateBERT --output-file <path_to_output_file.txt> --openai_api_key <your_api_key>
```

## Write your own demonstrations

In the [demonstrations](./demonstrations/) directory, you can find the demonstrations we have used to generate the dataset, which will help you in writing your own. Notice that the demonstration files are one sentence per line, and each targets the same group within each file. Once you've written the demonstrations and want to turn them into prompts, you can run this command:

```
python make_prompts.py --input_demonstrations <path_to_demo_file.txt> --output-file <path_to_prompt.txt> --demonstrations_per_prompt 5 --num_prompt_to_generate 100
```

## Using checkpoints of pretrained classifiers on ToxiGen

We have finetuned two toxicity detection classifiers on the ToxiGen data which has resulted in significant performance improvement as reported in the paper. The checkpoints for them can be loaded directly using the Huggingface's transformers library:

### HateBERT_ToxiGen

HateBERT finetuned on ToxiGen can be downloaded as follows in python:

```
from transformers import pipeline

toxigen_hatebert = pipeline("text-classification", model="thartvigsen/hatebert_toxigen/")

toxigen_hatebert(["I love science."])
```

### RoBERTa_ToxiGen

RoBERTa finetuned on ToxiGen can be downloaded as follows in python:

```
from transformers import pipeline

toxigen_roberta = pipeline("text-classification", model="thartvigsen/roberta_toxigen/")

toxigen_roberta(["I love science."])
```

#### Citation
Please use the following to cite this work:
```
@inproceedings{hartvigsen2022toxigen,
  title={ToxiGen: Controlling Language Models to Generate Implied and Adversarial Toxicity},
  author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```
