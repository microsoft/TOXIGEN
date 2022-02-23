# TOXIGEN: Controlling Language Models to Generate Implicit and Adversarial Toxicity

This is the official repository for [ToxiGen: Controlling Language Models to Generate Implicit and Adversarial Toxicity](https://arxiv.org/abs/2030.12345). 
ToxiGen is a massive dataset containing **implicitly** toxic and benign sentences mentioning minority groups. Classifiers trained on ToxiGen learn to detect notoriously-hard, subtle toxicity that includes no slurs or profanity.

The ToxiGen dataset is generated using large language models, and we also provide code for generating more data using your own language model.

This repository includes **two methods for generating new sentences with language models**:
- **Demonstration-Based Prompting**, where we encourage a language model to imitate hand-written examples
- **ALICE**, which tricks a predefined toxicity classifier into generating challenging, implicit sentences.

## Download the data

[Download the ToxiGen dataset](www.google.com). We provide the full training dataset, including some metadata about the prompts, and results for our human evaluation study.

## Generate new text with ToxiGen demonstration-based prompts

To pass our prompts into GPT-3 and generate new sentences, run this command:

```
python generate.py --input_prompt_file <path_to_prompt_file.txt> --language_model GPT3 --output_file <path_to_output_file.txt> --num_generations_per_prompt 10 --openai_api_key <your_api_key>
```

You can choose from a list of [prompt files](./prompts/) that we use in the paper or write your own and point to the file (shown below). A prompt file is a text file with one line per prompt (a string).

## Generate new text with ToxiGen demonstration-based prompts and ALICE

If you want to generate new text using ALICE, you need to also choose a pre-trained hate classifier. We provide a few here and some guidance on adding new classifiers (it's pretty straightforward). To generate with ALICE, run this command:

```
python generate.py --input_prompts <path_to_prompt_file.txt> --language_model GPT3 --ALICE True --classifier HateBERT --output-file <path_to_output_file.txt> --openai_api_key <your_api_key>
```

## Writing your own demonstrations

A boon of our method is that if you want to generate new text for a new group, you just need to write some new examples of the sorts of text you want to generate! In the [demonstrations](./demonstrations/) directory, you can find the demonstrations we used to generate ToxiGen, which may guide you in writing your own. Notice that the demonstration files are one sentence per line, and each targets the same group within each file. Once you've written some demonstrations and want to turn them into prompts, you can run this command:

```
python make_prompts.py --input_demonstrations <path_to_demo_file.txt> --output-file <path_to_prompt.txt> --demonstrations_per_prompt 5 --num_prompt_to_generate 100
```
