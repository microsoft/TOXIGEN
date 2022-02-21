from language_models import GPT3, ALICE
from pretrained_classifiers import HateBERT, RoBERTa
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prompt_file", type=str)
    parser.add_argument("--language_model", type=str)
    parser.add_argument("--classifier", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_generations_per_prompt", type=int)

    args = parser.parse_args()

    # --- load prompts ---
    prompts = [l.strip() for l in open(args.input_prompt_file).readlines()]

    # --- initialize language model ---
    if args.language_model == "GPT3":
        language_model = GPT3(apikey="<your_openai_api_key>")
    else:
        raise ValueError

    # --- initialize pretrained toxicity classifier ---
    if args.classifier == "HateBERT":
        classifier = HateBERT("<path_to_hatebert_files>")
    elif args.classifier == "RoBERTa":
        classifier = RoBERTa()

    # --- wrap language model and toxicity detector in ALICE ---
    adversarial_generator = ALICE(language_model, classifier)

    # --- loop through prompts and generate responses ---
    for prompt in prompts:
        for i in range(args.num_generations_per_prompt):
            response = adversarial_generator(prompt)
            with open(args.output_file, "a") as f:
                f.write(response)
