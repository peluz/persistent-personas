from datasets import load_dataset, Dataset, concatenate_datasets
import json

def load_data(dataset):
    roles = json.load(open("data/roles.json", "r"))
    if dataset == "instruction_role_specific":
        prompts = {}
        data = {}
        all_data = {}
        for role in roles.keys():
            if role == "empty": continue
            role_data = load_dataset(
            "json",
            data_files=f"https://huggingface.co/datasets/ZenMoore/RoleBench/raw/main/instructions-eng/role-specific-{role}.jsonl",
        )["train"]
            prompts[role] = role_data["instruction"]
            data[role] = role_data
            all_data[role] = role_data
        role_data = concatenate_datasets(list(all_data.values()))
        def remove_role(x):
            x["instruction"] = x["instruction"].split(", ", maxsplit=1)[-1]
            x["instruction"] = x["instruction"][0].upper() + x["instruction"][1:]
            return x
        role_data = role_data.map(remove_role)
        data["empty"] = role_data
        prompts["empty"] = role_data["instruction"]
        
    elif dataset == "instruction_general":
        prompts = {}
        data = {}
        all_data = load_dataset(
        "json",
        data_files=f"https://huggingface.co/datasets/ZenMoore/RoleBench/resolve/main/rolebench-eng/instruction-generalization/general/test.jsonl",
        )["train"].filter(lambda x: x["role"] in roles.keys())
        for role in roles.keys():
            role_data = all_data.filter(lambda x: x["role"] == role) if role != "empty" else Dataset.from_pandas(all_data.to_pandas().drop_duplicates("question"), preserve_index=False)
            prompts[role] = role_data["question"]
            data[role] = role_data

                
    # elif dataset == "alpaca-eval":
    #     data = load_dataset(
    #         "json",
    #         data_files="https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval_gpt4_baseline.json",
    #     )["train"]
    #     prompts = data["instruction"]
        
    elif dataset == "ifbench":
        data = load_dataset("allenai/IFBench_test")["train"]
        prompts = data["prompt"]
    elif dataset == "xstest":
        data = load_dataset("walledai/XSTest")["test"]
        prompts = data["prompt"]
    elif dataset == "bfi":
        personalities = json.load(open("data/characters_labels.json", "r"))
        roles = personalities.keys()
        data = json.load(open("data/BFI.json", "r"))
        prompts = [x["rewritten_en"] for x in data["questions"].values()]
    return data, prompts