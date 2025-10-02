import argparse
import json
import re
import spacy
import glob
import os
from pypremise import Premise, data_loaders
from data.loader import load_data
import pandas as pd

# Only run this if spacy is not already downloaded
# ! python -m spacy download en_core_web_sm

datasets = {
    "instruction_role_specific": load_data("instruction_role_specific")[1],
    "instruction_general": load_data("instruction_general")[1],
}

def run_experiment(group_0_sample, group_1_sample, file_name):
    nlp = spacy.load("en_core_web_sm")
    docs_0 = nlp.tokenizer.pipe(group_0_sample)
    docs_1 = nlp.tokenizer.pipe(group_1_sample)
    tokenized_group_0 = [[token.text for token in doc] for doc in docs_0]
    tokenized_group_1 = [[token.text for token in doc] for doc in docs_1]
    label_group_0 = [0 for _ in tokenized_group_0]
    label_group_1 = [1 for _ in tokenized_group_1]
    features = tokenized_group_0 + tokenized_group_1
    labels = label_group_0 + label_group_1
    premise_instances, voc_token_to_index, voc_index_to_token = (
        data_loaders.from_token_lists_and_labels(features, labels)
    )
    premise = Premise(voc_index_to_token=voc_index_to_token)
    premise_patterns = premise.find_patterns(premise_instances)
    patterns, group, count_0, count_1 = [], [], [], []
    for p in premise_patterns:
        patterns += [[x[0] for x in p.pattern.clauses]]
        group += [p.direction.value]
        count_0 += [p.label0_count]
        count_1 += [p.label1_count]
    df = pd.DataFrame(
        {"pattern": patterns, "group": group, "count_0": count_0, "count_1": count_1}
    )

    df.to_csv(file_name, index=False)
    print(f"Pattern has been saved to {file_name}.")


def extract_patterns(file_name):
    pattern_list = []
    with open(file_name, "r") as file:
        patterns = file.readlines()
    for pattern in patterns:
        extracted_content = re.findall(r"\((.*?)\)", pattern)
        pattern = extracted_content[:-1]
        pattern_list.append(pattern)
    total_patterns = [list(pair) for pair in set(tuple(pair) for pair in pattern_list)]
    return total_patterns




def get_samples(model, role, dataset, round, prefix):
    all_files = glob.glob(f"./generations/{model}/{prefix}{dataset}*_{round}.json")
    role_files = [x for x in all_files if role.replace(" ", "_") in x]
    no_role_files = [x for x in all_files if "empty" in x]
    print(f"Role files: {role_files}")
    print(f"No role files: {no_role_files}")
    role_samples = []
    no_role_samples = []
    for file in role_files:
        role_samples.extend(json.load(open(file, "r")))
    for file in no_role_files:
        no_role_samples.extend(json.load(open(file, "r")))
    role_samples = [x if x is not None else "null" for x in role_samples]
    no_role_samples = [x if x is not None else "null" for x in no_role_samples]
    if dataset in ["instruction_general", "instruction_role_specific"]:
        all_inputs  = datasets[dataset]
        empty = pd.DataFrame(list(all_inputs["empty"])).reset_index()
        inputs = all_inputs[role]
        if dataset == "instruction_role_specific":
            filtered_inputs = [
                x.split(", ", maxsplit=1)[-1] for x in inputs
            ]
            filtered_inputs = [
                x[0].upper() + x[1:] for x in filtered_inputs
            ]
            inputs_df = pd.DataFrame(filtered_inputs)
        else:
            inputs_df = pd.DataFrame(list(inputs))
        indices = pd.merge(
        inputs_df, empty.drop_duplicates(0), how="inner", on=0
    )["index"].values
        no_role_samples = [no_role_samples[i] for i in indices]
    return role_samples, no_role_samples


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment with a specified model and role."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-27b-it",
        help="The model to use for the experiment.",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="Michael Scott",
        help="The role to analyze in the experiment.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bfi",
        help="The dataset to use for the experiment.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        choices=[0, 102],
        help="The round number of the generations.",
    )

    parser.add_argument(
        "--dialogue",
        type=str,
        default="interview",
        choices=["interview", "instructions"],
        help="The dialogue type for the generations.",
    )
    args = parser.parse_args()

    # The model and role are now taken from command-line arguments
    model = args.model
    role = args.role
    dataset = args.dataset
    round = args.round
    dialogue = args.dialogue
    suffix = "" if round == 0 else "_last_round"
    prefix = "" if dialogue == "interview" else "inst-"
    file_name = (
        f"./results/premise/premise_{model}_{prefix}{dataset}_{role.replace(' ', '_')}{suffix}.csv"
    )
    if os.path.exists(file_name):
        print(f"The file '{file_name}' already exists. Skipping experiment.")
        return  # Exit the function if the file exists

    group_1_sample, group_0_sample = get_samples(model, role, dataset, round, prefix)
    run_experiment(group_0_sample, group_1_sample, file_name)


if __name__ == "__main__":
    main()
