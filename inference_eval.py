import argparse
from data.prompts import (
    flow_judge_template,
    selene_template_pairwise,
    selene_template_binary,
    selene_template_likert,
    prompt_map,
)
from data.loader import load_data
from data.constants import DATASETS
from pathlib import Path
from utils.seeds import initialize_seeds
from vllm.sampling_params import SamplingParams
import json
import glob
import os
import pandas as pd
import random
import numpy as np
import re


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(llm, model, judge, datasets, temperature, top_p, reference):
    initialize_seeds()
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=temperature,
        top_p=top_p,
        truncate_prompt_tokens=8192,
    )
    roles = json.load(open("data/roles.json", "r"))
    think_regex = re.compile(r'<think>.*?</think>\n*', re.DOTALL)

    for dataset in datasets:
        if dataset not in ["instruction_role_specific", "instruction_general"] and reference != "dataset":
            print(
                f"Skipping {dataset}: non-dataset references only supported for instruction_general and instruction_role_specific."
            )
            continue
        if "Selene" not in judge:
            judge_template = flow_judge_template
        else:
            if dataset in ["instruction_general", "instruction_role_specific"]:
                judge_template = selene_template_pairwise
            elif dataset == "xstest":
                judge_template = selene_template_binary
            else:
                judge_template = selene_template_likert

        generations = glob.glob(f"./generations/{model}/{dataset}*") + glob.glob(f"./generations/{model}/inst-{dataset}*")
        data, inputs = load_data(dataset)
        if dataset in ["instruction_role_specific", "instruction_general"]:
            all_inputs = inputs
        for generation in generations:
            if reference != "dataset":
                if "_0.json" in generation:
                    continue
                if reference == "previous":
                    all_steps = glob.glob(re.sub(r"\d+\.json", "*.json", generation))
                    sorted_steps = sorted(
                        all_steps,
                        key=lambda x: int(
                            re.search(r"(?<=_)\d+(?=\.json)", x).group(0)
                        ),
                    )
                    current_idx = sorted_steps.index(generation)
                    previous_step = sorted_steps[current_idx - 1]
                    references = previous_step
                    suffix = "_previous_reference"
                    print(
                        f"Using previous step {previous_step} as reference for {generation}."
                    )
                elif reference == "zero":
                    zero_generation = re.sub(r"(\d+\.json)", "0.json", generation)
                    references = zero_generation
                    suffix = "_zero_reference"
                    print(
                        f"Using zero step {zero_generation} as reference for {generation}."
                    )
                reference_inferences = json.load(open(references, "r"))
            else:
                suffix = ""
            inferences = json.load(open(generation, "r"))
            role = (
                re.search(rf"(?<={dataset}_)[^.]*", generation)
                .group(0)
                .rsplit("_", maxsplit=1)[0]
                .replace("_", " ")
            )
            if role == "empty" and dataset in [
                "instruction_role_specific",
                "instruction_general",
            ]:
                reference_roles = [x for x in roles.keys() if x != "empty"]
            else:
                reference_roles = [role]
            for r_role in reference_roles:
                result_path = (
                    Path(
                        f"./metrics/{dataset}/{judge.split('/')[-1]}/{model}/{Path(generation).name}{suffix}.csv"
                    )
                    if role != "empty"
                    or dataset
                    not in ["instruction_role_specific", "instruction_general"]
                    else Path(
                        f"./metrics/{dataset}/{judge.split('/')[-1]}/{model}/{Path(generation).name}-{r_role}.csv"
                    )
                )
                if dataset in ["instruction_role_specific", "instruction_general"]:
                    inputs = all_inputs[r_role]
                if result_path.exists():
                    print(
                        f"Skipping generation {generation} from model {model} using judge {judge} for task {dataset} and role {r_role}: already computed."
                    )
                    continue
                print(
                    f"Evaluating generation {generation} from model {model} using judge {judge} for task {dataset} and role {r_role}."
                )
                role_description = roles[r_role]
                preamble = (
                    f"""<role>
{r_role}
</role>
<role_description>
{role_description}
</role_description>"""
                    if r_role != "empty"
                    else ""
                )
                if role != "empty" or dataset not in [
                    "instruction_role_specific",
                    "instruction_general",
                ]:
                    outputs = inferences
                else:
                    empty = pd.DataFrame(list(all_inputs["empty"])).reset_index()
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
                    outputs = list(np.array(inferences)[indices])

                prompts = []
                if dataset == "bfi":
                    dimensions = [x["dimension"] for x in data["questions"].values()]
                for idx, (i, o) in enumerate(zip(inputs, outputs)):
                    if o is None: o = "null" # Sometimes gemini returns None due to safety filters
                    metric = prompt_map[dataset]
                    if dataset == "bfi":
                        metric = metric[dimensions[idx]]
                    if dataset == "instruction_general":
                        if reference == "dataset":
                            reference_answer = data[r_role]["generated"][idx][0]
                        else:
                            reference_answer = reference_inferences[idx]
                    elif dataset == "instruction_role_specific":
                        if reference == "dataset":
                            reference_answer = data[r_role]["answer"][idx]
                        else:
                            reference_answer = reference_inferences[idx]
                    if dataset in ["instruction_general", "instruction_role_specific"]:
                        response_a = reference_answer
                        response_b = o.rstrip(" _\n")
                        if idx % 2 != 0:
                            response_a, response_b = response_b, response_a
                        prompt = metric(
                            f"{preamble}\n<user_query>\n{i}\n</user_query>",
                            response_a=response_a,
                            response_b=response_b,
                            judge_template=judge_template,
                        )
                    else:
                        prompt = metric(
                            f"{preamble}\n<user_query>\n{i}\n</user_query>",
                            output=think_regex.sub("", o).rstrip(" _\n"),
                            judge_template=judge_template,
                        )
                    prompts.append([{"role": "user", "content": prompt}])
                print(f"Example prompt: {random.choice(prompts)[-1]['content']}")
                llm.chat(prompts[0], sampling_params=sampling_params)
                outputs = llm.chat(prompts, sampling_params=sampling_params)
                responses = []
                for output in outputs:
                    generated_text = output.outputs[0].text
                    responses.append(generated_text)
                result_path.parent.mkdir(exist_ok=True, parents=True)
                pd.DataFrame({dataset: responses}).to_csv(result_path, index=False)


if __name__ == "__main__":
    import vllm

    parser = argparse.ArgumentParser(
        description="Get interview scores for the roles of model {model} using judge model {judge}."
    )
    parser.add_argument("model", help="The model  to be prompted.", type=str)
    parser.add_argument("judge", help="The judge model to be prompted.", type=str)
    parser.add_argument(
        "--datasets",
        help="The datasets to be used for the inference.",
        type=str,
        nargs="+",
        default=DATASETS[1:],
    )
    parser.add_argument("--gpus", help="Number of gpus", type=int, default=1)
    parser.add_argument(
        "--temperature",
        help="Temperature for probabiliy scaling.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top_p",
        help="Top-p proability of tokens for nucleus sampling",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dtype", help="dtype to load the model", type=str, default="auto"
    )
    parser.add_argument(
        "--reference",
        type=str,
        choices=["dataset", "zero", "previous"],
        default="dataset",
        help="Which response to use reference answer for instruction general and role specific.",
    )
    args = parser.parse_args()
    llm = vllm.LLM(
        model=args.judge,
        enable_prefix_caching=True,
        dtype=args.dtype,
        tensor_parallel_size=args.gpus,
        #   download_dir=os.environ["HF_MODELS"],
        gpu_memory_utilization=0.95,
        disable_cascade_attn=True,
    )
    main(
        llm,
        args.model,
        args.judge,
        args.datasets,
        args.temperature,
        args.top_p,
        args.reference,
    )
