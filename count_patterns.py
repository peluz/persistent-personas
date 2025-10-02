import json
import ast
import spacy
import pandas as pd
import numpy as np
import glob
import os
from multiprocessing import Pool, cpu_count
from data.loader import load_data

from tqdm.auto import tqdm
from data.constants import DATASETS

# Load spacy once globally to be available for each process
nlp = spacy.load("en_core_web_sm")

all_datasets = {
    "instruction_role_specific": load_data("instruction_role_specific")[1],
    "instruction_general": load_data("instruction_general")[1],
}

def get_pattern_counts(args):
    """
    Wrapper function to get pattern counts for a single trial.
    This function takes a single tuple of arguments to be used with Pool.map.
    """
    setting, model, dataset, role = args
    prefix = "inst-" if setting == "goal-oriented" else ""
    role_file_name = role.replace(" ", "_")

    try:
        df = pd.read_csv(f"./results/premise/premise_{model}_{dataset}_{role_file_name}.csv", converters={'pattern': ast.literal_eval})
    except FileNotFoundError:
        print(f"Skipping {model}-{dataset}-{role} (premise file not found)")
        return pd.DataFrame()

    samples_role, samples_no_role = [], []
    for i in np.linspace(0, 102, 10):
        # Read the persona-assigned generations
        role_file = f"generations/{model}/{prefix}{dataset}_{role_file_name}_{int(i)}.json"
        try:
            role_gens = [x if x is not None else "null"for x in json.load(open(role_file, "r"))]
            role_docs = nlp.tokenizer.pipe(role_gens)
            samples_role.append([[token.text for token in doc] for doc in role_docs])
        except FileNotFoundError:
            samples_role.append([])

        # Read the control (no-role) generations
        no_role_file = f"generations/{model}/{prefix}{dataset}_empty_{int(i)}.json"
        try:
            no_role_gens = [x if x is not None else "null" for x in json.load(open(no_role_file, "r"))]
            if dataset in ["instruction_general", "instruction_role_specific"]:
                all_inputs  = all_datasets[dataset]
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
                no_role_gens = [no_role_gens[i] for i in indices]
            no_role_docs = nlp.tokenizer.pipe(no_role_gens)
            samples_no_role.append([[token.text for token in doc] for doc in no_role_docs])
        except FileNotFoundError:
            samples_no_role.append([])

    data = []
    turns = np.linspace(0, 102, 10)
    for i, (docs_role, docs_no_role) in enumerate(zip(samples_role, samples_no_role)):
        turn = int(turns[i])
        for idx, pattern in enumerate(df.pattern.tolist()):
            pattern_group = df.group.iloc[idx]
            
            if docs_role:
                count_role_gen = np.sum([set(pattern).issubset(set(x)) for x in docs_role])
            else:
                count_role_gen = 0
            
            if docs_no_role:
                count_no_role_gen = np.sum([set(pattern).issubset(set(x)) for x in docs_no_role])
            else:
                count_no_role_gen = 0

            data.append({
                'model': model,
                'dataset': dataset,
                'role': role,
                'setting': setting,
                'turn': turn,
                'pattern_group': pattern_group,
                'pattern': str(pattern),
                'count_in_role_gen': count_role_gen,
                'count_in_no_role_gen': count_no_role_gen
            })

    return pd.DataFrame(data)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    """
    Main loop to run the analysis for all combinations using multiprocessing.
    """
    models = [x.split("/")[-1] for x in glob.glob("./generations/*")]
    datasets = DATASETS
    roles = [x for x in json.load(open("data/roles.json", "r")).keys() if x != "empty"]
    settings = ["persona-directed", "goal-oriented"]

    # Create a list of all possible argument combinations
    args_list = []
    for setting in settings:
        for model in models:
            for dataset in datasets:
                for role in roles:
                    args_list.append((setting, model, dataset, role))

    # Determine the number of processes to use. It's often best to use
    # a few less than the total number of cores to avoid maxing out the CPU.
    try:
        num_processes = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        # Fallback for running locally without Slurm
        num_processes = os.cpu_count()
    print(f"Using {num_processes} processes for parallel execution.")

    all_counts_dfs = []
    with Pool(processes=num_processes) as pool:
        # The tqdm library wraps the pool.imap_unordered iterator to display a progress bar.
        # It's an efficient way to get real-time updates.
        for result_df in tqdm(pool.imap_unordered(get_pattern_counts, args_list), total=len(args_list), desc="Processing combinations"):
            if not result_df.empty:
                all_counts_dfs.append(result_df)

    # Concatenate the results from all processes
    master_df = pd.concat([df for df in all_counts_dfs if not df.empty], ignore_index=True)
    
    if not master_df.empty:
        print("Master DataFrame created. Shape:", master_df.shape)
        master_df.to_csv("./results/all_pattern_counts.csv", index=False)
        print("Master DataFrame saved to ./results/all_pattern_counts.csv")
    else:
        print("No data was collected.")

if __name__ == "__main__":
    main()
