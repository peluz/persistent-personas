import argparse
from data.prompts import *
from pathlib import Path
from utils.seeds import initialize_seeds
from tqdm.auto import tqdm
from vllm.sampling_params import SamplingParams
import json
import glob
import os
import pandas as pd
import random
import re


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(llm, model, judge, temperature, top_p, conversation):
    initialize_seeds()
    sampling_params = SamplingParams(max_tokens=8192, temperature=temperature)
    roles = json.load(open("data/roles.json", "r"))
    interviews = glob.glob(f"./generations/{model}/{conversation}*")
    think_regex = re.compile(r'<think>.*?</think>\n*', re.DOTALL)
    judge_template = (
        selene_template_likert if "Selene" in judge else flow_judge_template
    )

    for interview in interviews:
        dialogue = json.load(open(interview, "r"))

        for metric in [knowledge_prompt, style_prompt, in_character_prompt]:
            metric_name = metric.__name__.replace("_prompt", "")
            role = (
                re.search(rf"(?<={conversation}_).*(?=\.json)", interview)
                .group(0)
                .replace("_shuffle", "")
                .replace("_", " ")
            )
            if role == "empty":
                reference_roles = [x for x in roles.keys() if x != "empty"]
            else:
                reference_roles = [role]
            for r_role in reference_roles:
                result_path = (
                    Path(
                        f"./metrics/{metric_name}/{judge.split('/')[-1]}/{model}/{Path(interview).name}.csv"
                    )
                    if role != "empty"
                    else Path(
                        f"./metrics/{metric_name}/{judge.split('/')[-1]}/{model}/{Path(interview).name}-{r_role}.csv"
                    )
                )
                if result_path.exists():
                    print(
                        f"Skipping interview {interview} from model {model} using judge {judge} and metric {metric_name} for role {r_role}: already computed."
                    )
                    continue
                print(
                    f"Evaluating interview {interview} from model {model} using judge {judge} and metric {metric_name} and role {r_role}."
                )
                role_description = roles[r_role]
                preamble = f"""<role>
{r_role}
</role>
<role_description>
{role_description}
</role_description>
"""
                inputs = [x["content"] for x in dialogue[0::2]]
                outputs = [x["content"] for x in dialogue[1::2]]
                prompts = []
                for i, o in zip(inputs, outputs):
                    prompt = metric(
                        f"{preamble}<user_query>\n{i}\n</user_query>",
                        output=think_regex.sub("", o),
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
                pd.DataFrame({metric_name: responses}).to_csv(result_path, index=False)


if __name__ == "__main__":
    import vllm

    parser = argparse.ArgumentParser(
        description="Get interview scores for the roles of model {model} using judge model {judge}."
    )
    parser.add_argument("model", help="The model model to be prompted.", type=str)
    parser.add_argument("judge", help="The judge model to be prompted.", type=str)
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
        "--conversation",
        type=str,
        default="interview",
        help="Which generated conversation to use",
        choices=["interview", "instructions"],
    )
    args = parser.parse_args()
    llm = vllm.LLM(
        model=args.judge,
        enable_prefix_caching=True,
        dtype=args.dtype,
        tensor_parallel_size=args.gpus,
        #   download_dir=os.environ["HF_MODELS"],
        gpu_memory_utilization=0.95,
    )
    main(llm, args.model, args.judge, args.temperature, args.top_p, args.conversation)
