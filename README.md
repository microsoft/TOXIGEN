# [ToxiGen](http://arxiv.org/abs/2203.09509): A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection ![Github_Picture](https://user-images.githubusercontent.com/13631873/159418812-98ccfe19-1a63-4bc9-9692-92f096f443b6.png)

## [June 17, 2024] Update: Releasing 27,450 human annotations.
You can now download the raw human annotations via `load_dataset("toxigen/toxigen-data", "annotations")`. These data include 27,450 responses from all mechanical turk annotations. All WorkerIDs have been hashed to further anonymize the annotators.

## Overview
This repository includes all necessary components that we used to generate ToxiGen dataset which contains implicitly toxic and benign sentences mentioning 13 minority groups. It includes a tool referred to as ALICE to stress test a given off-the-shelf content moderation system and iteratively improve it across these minority groups.

With release of the source codes and prompt seeds for this work we hope to encourage and engage community to contribute to it by for example adding prompt seeds and generating data for minority groups that are not covered in our dataset or even scenarios we have not covered to continuously iterate and improve it (e.g., by submitting PR to this repository). 

The dataset is intended to be used for training classifiers that learn to detect subtle hate speech that includes no slurs or profanity. The data, methods and two trained hatespeech detection checkpoints released with this work are intended to be used for research purposes only. 

This repository includes two methods for generating new sentences given a large scale pretrained language model (e.g., GPT3) and an off the shelf classifier:
- **Demonstration-Based Prompting**, where a language model is used to create more data given human provided prompts across different minority groups. 
- **ALICE**, which creates an adversarial set up between a given toxicity classifier and a generator (pretrained language model) to create challenging examples for the classifier and improve its performance. 

**WARNING: This repository contains and discusses content that is offensive or upsetting. All materials are intended to support research that improves toxicity detection methods. Included examples of toxicity do not represent how the authors or sponsors feel about any identity groups.**

## Downloading ToxiGen

ToxiGen is available on [HuggingFace](https://huggingface.co/datasets/toxigen/toxigen-data).

To download with python, you'll need to create a Hugging Face auth_token by following [these instructions](https://huggingface.co/docs/hub/security-tokens). As discussed below, you can manually use `use_auth_token={auth_token}` or register your token with your transformers installation via huggingface-cli.

```
from datasets import load_dataset
train_data = load_dataset("toxigen/toxigen-data", name="train", use_auth_token=True) # 250k training examples
annotated_data = load_dataset("toxigen/toxigen-data", name="annotated", use_auth_token=True) # Human study
raw_annotations = load_dataset("toxigen/toxigen-data", name="annotations", use_auth_token=True) # Raw Human study
```

**Optional, but helpful**: Please fill out [this form](https://forms.office.com/r/r6VXX8f8vh) so we can track how the community uses ToxiGen.

### Authorization Tokens

There are two ways to obtain authorization tokens (for which you will need a huggingface account):
1. Follow [these directions](https://huggingface.co/docs/hub/security-tokens) to pass an authorization token in while loading the data.
2. Use [huggingface-cli](https://huggingface.co/docs/transformers/model_sharing#setup), as you could when sharing a model on HuggingFace, to associate your huggingface account with your installed version of the transformers library.

## Installing ToxiGen source code

ToxiGen is bundled into a python package that can be installed using pip:
```
pip install toxigen
```

## Jupyter Notebook Example
Please use this [Jupyter Notebook](./notebooks/generate_text.ipynb) to get started with the main components of this repository.

We also include a [Notebook](./notebooks/load_datasets.ipynb) showing how to download different parts of ToxiGen using HuggingFace.

## Generating data with ToxiGen demonstration-based prompts
To generate data by passing prompts into the pretrained language model (GPT-3) used in this work please use the following command:

```
python generate.py --input_prompt_file <path_to_prompt_file.txt> --language_model GPT3 --output_file <path_to_output_file.txt> --num_generations_per_prompt 10 --api_key <your_api_key>
```

You can choose from a list of [prompt files](./prompts/) that we have used in this work or write your own and point to the file (shown below). A prompt file is a text file with one line per prompt (a string).

## Generating data using ALICE

To generate data using ALICE, it is necessary to choose a generator (GPT3 in our case) and a pre-trained hate speech classifier. We provide examples here and the guidance about how to add new classifiers. To generate with ALICE, run this command:

```
python generate.py --input_prompt_file <path_to_prompt_file.txt> --language_model GPT3 --ALICE True --classifier HateBERT --output_file <path_to_output_file.txt> --api_key <your_api_key>
```

## Writing your own demonstrations

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

toxigen_hatebert = pipeline("text-classification", model="tomh/toxigen_hatebert", tokenizer="bert-base-uncased")

toxigen_hatebert("I love science")
```

or

```
from transformers import AutoModelForSequenceClassification

toxigen_hatebert = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_hatebert")
```

### RoBERTa_ToxiGen

RoBERTa finetuned on ToxiGen can be downloaded as follows in python:

```
from transformers import pipeline

toxigen_roberta = pipeline("text-classification", model="tomh/toxigen_roberta")

toxigen_roberta("I love science")
```
or

```
from transformers import AutoModelForSequenceClassification

toxigen_hatebert = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")
```

### Contributing
We encourage contribution to the ToxiGen repository of prompts and demonstrations. If you find your new prompts that work for your cases, please add them.

#### Community Contributions
*March 9, 2024*: Demonstrations for Immigrants and Bisexuality were added from a Zurich hackathon.

### Citation
Please use the following to cite this work:
```
@inproceedings{hartvigsen2022toxigen,
  title={ToxiGen: A Large-Scale Machine-Generated Dataset for Implicit and Adversarial Hate Speech Detection},
  author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```

### Responsible AI Considerations
Please also note that there is still a lot that this dataset is not capturing about what constitutes problematic language. Our annotations might not capture the full complexity of these issues, given problematic language is context-dependent, dynamic, and can manifest in different forms and different severities. Problematic language is also fundamentally a human-centric problem and should be studied in conjunction with human experience. There is need for multi-disciplinary work to better understand these aspects. Also note that this dataset only captures implicit toxicity (more precisely hate speech) for 13 identified minority groups, and due to its large scale can naturally be noisy. Our goal in this project is to provide the community with means to improve toxicity detection on implicit toxic language for the identified minority groups and there exists limitations to this dataset and models trained on it which can potentially be the subject of future research, for example, including more target groups, a combination of them and so on that are not covered in our work.
