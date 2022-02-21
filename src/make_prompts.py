import argparse
import numpy as np

def write_and_shuffle_prompts(list_of_strings: list, num_sentences):
    """ 
    Given a list of prompt sentences, create an engineered prompt
    Example:
        list_of_strings = ["string1", "string2", "string3"]
        >>> write_and_shuffle_prompts(list_of_strings)
        "- string2\n- string1\n- string3\n-"
    """ 
    if num_sentences > len(list_of_strings):
        num_sentences = len(list_of_strings)
    list_of_strings = [list_of_strings[i] for i in np.random.choice(range(len(list_of_strings)), num_sentences, replace=False)]
    prompt = ""
    for s in list_of_strings:
        prompt += "- {}".format(s.replace("'", ""))
    prompt += "-"
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_demonstrations", type=str)
    parser.add_argument("--demonstrations_per_prompt", type=int, default=5)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_prompts_to_generate", type=int, default=100)

    args = parser.parse_args()

    with open(args.input_demonstrations, "r") as f:
        text = f.readlines()

    print(f"Writing prompts to {args.output_file}")
    for i in range(args.num_prompts_to_generate):
        with open(f"{args.output_file}", "a") as f:
            prompt = write_and_shuffle_prompts(text, args.demonstrations_per_prompt)
            f.write(prompt.__repr__()[1:-1])
            f.write("\n")

if __name__ == "__main__":
    main()
