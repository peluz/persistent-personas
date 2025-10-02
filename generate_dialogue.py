import argparse
import json
from data.prompts import role_template
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from utils.seeds import initialize_seeds
from models.gemini import call_gemini_with_retry
from random import shuffle
from pathlib import Path
from google import genai
from google.genai import types
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(
        description="Generate dialogue using the CAMEL library with vLLM as the inference engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Arguments for Hugging Face model IDs
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face model ID for the assistant simulator (e.g., 'meta-llama/Llama-2-7b-chat-hf').",
    )

    parser.add_argument(
        "--role", type=str, required=True, help="Persona to be role-played."
    )

    # Arguments for CAMEL dialogue generation (examples)
    parser.add_argument(
        "--num_dialogue_turns",
        type=int,
        default=300,
        help="Number of dialogue turns to generate.",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="interview.json",
        help="Which queries set to use",
        choices=["interview.json", "instructions.json"],
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="dialogue.txt",
        help="Path to the file to save the generated dialogue.",
    )

    # Arguments for vLLM (examples)
    parser.add_argument(
        "--vllm_host", type=str, default="localhost", help="Host for the vLLM server."
    )
    parser.add_argument(
        "--vllm_port", type=int, default=8000, help="Port for the vLLM server."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature for vLLM."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate per response.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow executing custom code for models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory to be allocated for the vLLM engine.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Always use eager-mode for model execution. If false, use torch.compile for faster inference.",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle interview questions."
    )

    args = parser.parse_args()

    print(f"Assistant Model ID: {args.model_id}")
    print(f"Role: {args.role}")
    print(f"Number of Dialogue Turns: {args.num_dialogue_turns}")
    print(f"Output File: {args.output_file}")
    print(f"Queries: {args.queries}")
    print(f"vLLM Host: {args.vllm_host}")
    print(f"vLLM Port: {args.vllm_port}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Trust Remote Code: {args.trust_remote_code}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Enforce Eager: {args.enforce_eager}")
    print(f"Shuffle: {args.shuffle}")

    initialize_seeds()
    suffix = "_shuffle" if args.shuffle else ""
    result_path = Path(
        f"generations/{args.model_id.split('/')[-1]}/{args.queries.replace('.json', '')}_{args.role.replace(' ', '_')}{suffix}.json"
    )
    if result_path.exists():
        print(
            f"Skipping generations from model {args.model_id} role-playing {args.role} for queries {args.queries}: already computed."
        )
        return

    if "gemini" in args.model_id:
        client = genai.Client()
    elif "gpt-oss" in args.model_id:
        client = OpenAI(
            base_url=f"http://{args.vllm_host}:{args.vllm_port}/v1",
            api_key="token-abc123",
        )
    else:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type=args.model_id,
            model_config_dict={"temperature": args.temperature, "top_p": 0.95, "max_tokens": args.max_tokens},
            url=f"http://{args.vllm_host}:{args.vllm_port}/v1",
        )
    roles = json.load(open("./data/roles.json"))
    role_desc = roles[args.role]

    sys_msg = role_template.format(role=args.role, role_desc=role_desc)
    if "Nemotron" in args.model_id:
        sys_msg = f"detailed thinking: off\n\n{sys_msg}"
    interview = json.load(open(f"./data/{args.queries}", "r"))
    questions = interview["questions"]
    if args.shuffle:
        shuffle(questions)
    role = args.role if args.role != "empty" else args.model_id.split("/")[-1]
    if "instructions" not in args.queries:
        queries = (
            [interview["introductory_remark"].replace("{role}", role)]
            + questions
            + [interview["concluding_remark"].replace("{role}", role)]
        )
    else:
        queries = questions
    if "gemini" in args.model_id:
        if args.role != "empty":
            assistant = client.chats.create(
                model=args.model_id,
                config=types.GenerateContentConfig(
                    system_instruction=sys_msg,
                    temperature=args.temperature,
                    top_p=0.95,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        else:
            assistant = client.chats.create(
                model=args.model_id,
                config=types.GenerateContentConfig(
                    temperature=args.temperature,
                    top_p=0.95,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
    elif "gpt-oss" not in args.model_id:
            if args.role != "empty":
                 assistant = ChatAgent(sys_msg, model=model, token_limit=9999999)
            else:
                if "Nemotron" in args.model_id:
                    assistant = ChatAgent("detailed thinking: off", model=model, token_limit=9999999)
                else:
                    assistant = ChatAgent(model=model, token_limit=9999999)
    chat = []
    if args.role != "empty" and "gpt-oss" in args.model_id:
        chat.append({"role": "developer", "content": sys_msg})
    for q in queries:
        chat += [{"role": "user", "content": q}]
        print(f"User: {q}")
        print()
        if "gemini" in args.model_id:
            response, assistant = call_gemini_with_retry(assistant, q)
            message = response.text
        elif "gpt-oss" in args.model_id:
            response = client.responses.create(
                input=chat,
                model=args.model_id,
                temperature=args.temperature,
                top_p=0.95,
                reasoning={"effort": "low"},
            )
            print(response)
            message = response.output_text
        else:
            response = assistant.step(q)
            message = response.msgs[0].content
        chat += [{"role": "assistant", "content": message}]
        print(f"Assistant: {message}")
        print()
    result_path.parent.mkdir(exist_ok=True, parents=True)
    json.dump(chat, open(result_path, "w"))


if __name__ == "__main__":
    main()
