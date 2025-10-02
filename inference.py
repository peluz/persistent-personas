import argparse
from data.prompts import *
from data.loader import load_data
from data.constants import DATASETS
from models.gemini import call_gemini_with_retry
from datasets import load_dataset
from pathlib import Path
from utils.seeds import initialize_seeds
from vllm.sampling_params import SamplingParams
from google import genai
from google.genai import types
from tqdm.auto import tqdm
import json
import numpy as np
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(llm, model, datasets, temperature, top_p, conversation):
    initialize_seeds()
    model = model.split("/")[-1]
    sampling_params = SamplingParams(max_tokens=8192, temperature=temperature)
    all_roles = json.load(open("data/roles.json", "r"))
    characters_labels = json.load(open("data/characters_labels.json", "r"))
    for dataset in datasets:
        _, data = load_data(dataset)
        # if dataset == "bfi":
        #     roles = characters_labels
        # else:
        #     roles = all_roles
        for role in all_roles.keys():
            interview = json.load(
                open(
                    f"./generations/{model}/{conversation}_{role.replace(' ', '_')}.json",
                    "r",
                )
            )
            for t in np.linspace(0, len(interview) / 2, 10):
                t = int(t)
                if conversation == "instructions":
                    prefix = "inst-"
                else:
                    prefix = ""
                result_path = Path(
                    f"./generations/{model}/{prefix}{dataset}_{role.replace(' ', '_')}_{t}.json"
                )
                if result_path.exists():
                    print(
                        f"Skipping generations from model {model} for role {role} and dataset {dataset} with base conversation {conversation}: already computed."
                    )
                    continue
                dialogue = interview[: t * 2]
                system = "developer" if "gpt-oss" in model else "system"
                system_message = role_template.format(
                    role=role, role_desc=all_roles[role]
                )
                if "Nemotron" in model:
                    system_message = f"detailed thinking: off\n\n{system_message}"
                print(
                    f"Generating responses for role {role} from model {model} on dataset {dataset} and position {t}."
                )
                responses = []
                if "gemini" not in model:
                    if dataset not in [
                        "instruction_role_specific",
                        "instruction_general",
                    ]:
                        if role != "empty":
                            prompts = [
                                [{"role": system, "content": system_message}]
                                + dialogue
                                + [{"role": "user", "content": q}]
                                for q in data
                            ]
                        else:
                            if "Nemotron" in model:
                                prompts = [
                                    [{"role": system, "content": "detailed thinking: off"}]
                                    + dialogue
                                    + [{"role": "user", "content": q}]
                                    for q in data
                                ]
                            else:
                                prompts = [
                                dialogue + [{"role": "user", "content": q}]
                                for q in data
                            ]
    
                    else:
                        if role != "empty":
                            prompts = [
                                    [{"role": system, "content": system_message}]
                                    + dialogue
                                    + [{"role": "user", "content": q}]
                                    for q in data[role]
                                ]
                        else:
                            if "Nemotron" in model:
                                prompts = [
                                    [{"role": system, "content": "detailed thinking: off"}]
                                    + dialogue
                                    + [{"role": "user", "content": q}]
                                    for q in data[role]
                                ]
                            else:
                                prompts = [
                                    dialogue + [{"role": "user", "content": q}]
                                    for q in data[role]
                                ]
                    print(f"Example prompt: {prompts[0]}")
                    llm.chat(prompts[0], sampling_params=sampling_params)
                    outputs = llm.chat(prompts, sampling_params=sampling_params)
                    for output in outputs:
                        generated_text = output.outputs[0].text.rstrip(" _\n")
                        responses.append(generated_text)
                else:
                    chat_history = []
                    for utterance in dialogue:
                        speaker = (
                            "model" if utterance["role"] == "assistant" else "user"
                        )
                        part = [{"text": utterance["content"]}]
                        chat_history.append({"role": speaker, "parts": part})
                    if dataset not in [
                        "instruction_role_specific",
                        "instruction_general",
                    ]:
                        queries = data
                    else:
                        queries = data[role]
                    print(f"Example prompt: {chat_history}\n{queries[0]}")
                    for q in tqdm(queries):
                        if role != "empty":
                            assistant = llm.chats.create(
                                model=model,
                                history=chat_history,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_message,
                                    temperature=temperature,
                                    top_p=0.95,
                                    thinking_config=types.ThinkingConfig(
                                        thinking_budget=0
                                    ),
                                ),
                            )
                        else:
                            assistant = llm.chats.create(
                                model=model,
                                history=chat_history,
                                config=types.GenerateContentConfig(
                                    temperature=temperature,
                                    top_p=0.95,
                                    thinking_config=types.ThinkingConfig(
                                        thinking_budget=0
                                    ),
                                ),
                            )
                        response, _ = call_gemini_with_retry(assistant, q)
                        responses.append(response.text)
                result_path.parent.mkdir(exist_ok=True, parents=True)
                json.dump(responses, open(result_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for model {model} on datasets {datasets}."
    )
    parser.add_argument("model", help="The model model to be prompted.", type=str)
    parser.add_argument(
        "--datasets",
        help="The datasets to be used for the inference.",
        type=str,
        nargs="+",
        default=DATASETS,
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
        "--conversation",
        type=str,
        default="interview",
        help="Which generated conversation to use",
        choices=["interview", "instructions"]
    )
    args = parser.parse_args()
    if "gemini" not in args.model:
        import vllm

        llm = vllm.LLM(
            model=args.model,
            enable_prefix_caching=True,
            dtype=args.dtype,
            trust_remote_code=True,
            tensor_parallel_size=args.gpus,
            max_model_len=128000,
            #   download_dir=os.environ["HF_MODELS"],
            gpu_memory_utilization=0.95,
            disable_cascade_attn=True,
        )
    else:
        llm = genai.Client()
    main(llm, args.model, args.datasets, args.temperature, args.top_p, args.conversation)
