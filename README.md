# Persistent personas
This repo holds source code for the paper [Persistent Personas: Role-Playing, Instruction Following, and Safety in Extended Interactions](https://arxiv.org/abs/2512.12775), to be presented at EACL 2026.


## Requirement

- [miniforge](https://github.com/conda-forge/miniforge)

## Setting up 

1. Run the snippet below to install all dependencies:

```console
conda env create -f environment.yml
```

2. Download BFI and character labels from https://github.com/Neph0s/InCharacter, save in data/BFI.json and data/character_labels.json

## Persona generations
- Generations from all personas for all models, datasets, and dialogue conditions are available in the "results" folder.


## Reproducing the experiments
- To generate dialogues, first serve VLLM model:
```console
python -m vllm.entrypoints.openai.api_server --model $MODEL_URL \
                                         --host $VLLM_HOST \
                                         --port $VLLM_PORT \
                                         --tensor-parallel-size $NUM_GPUS \
                                         --trust-remote-code  \
                                         --gpu-memory-utilization 0.95 \
					                     --max-model-len 128000 \

```
And then submit queries to the model:
```console
python -u generate_dialogue.py \
    --model_id "$MODEL_URL" \
    --role "$role" \ # The name of the persona in data/roles.json to be assigned
    --shuffle \ # Add this to shuffle the queries
    --temperature 0.0 \
    --vllm_port $VLLM_PORT \
    --queries {instructions.json or interview.json} # Goal-oriented or persona-directed queries

```


- Run the commands below to generate responses for dialogue conditioned datasets:
```console
python -u inference.py $MODEL_URL --gpus $NUM_GPUS --conversation interview
python -u inference.py $MODEL_URL --gpus $NUM_GPUS --conversation instructions
```

- Run commands below to evaluate dialogue utterances (knowledge, style, and in-character consistency)
```console
python -u interview_eval.py $MODEL_NAME AtlaAI/Selene-1-Mini-Llama-3.1-8B --conversation interview
python -u interview_eval.py $MODEL_NAME AtlaAI/Selene-1-Mini-Llama-3.1-8B --conversation instructions
```

- Run commands below to evaluate responses to dialogue-conditioned datasets
```console
python -u inference_eval.py $MODEL_NAME AtlaAI/Selene-1-Mini-Llama-3.1-8B
```

- Run commands below to extract persona and baseline token patterns for a given model-persona-dataset combination
```console
python -u extract_patterns.py --role "$ROLE" --dataset "$DATASET" --model "$MODEL" --dialogue instructions
python -u extract_patterns.py --role "$ROLE" --dataset "$DATASET" --model "$MODEL" --dialogue interview
```

- Run command below to count pattern occurrences
```console
python -u count_patterns.py
```

- Notebook extract_ratings.ipynb extracts judge ratings from the judge generations.
- Notebook gen_graphics.ipynb creates most of the figures presented in the table.
- Notebook jude_judges.ipynb documents judge validation.
- Notebook length_control.ipynb compares persona-directed and goal-oriented dialogues w.r.t. number of tokens.
- Notebook prism_sampling.ipynb documents how goal-oriented queries were sampled from PRISM.
- Notebook safety_analysis.ipynb documents the analysis of personas' responses to XSTest queries.
- Notebook spotlight_analysis.ipynb documents the analyses of token patterns.
