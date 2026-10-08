#!/bin/python3.13
import os, toml, re, json, sys, getopt
from jinja2 import Template
from copy import deepcopy
from tqdm import tqdm
from src.PathEvaluate import PathLM
from collections import Counter
import numpy as np

config = toml.load("config.toml")

def generate_probability(model_name, samples):
    model = PathLM(model_name)
    result = Counter()
    tracking = {}
    for key, value in tqdm(samples):
        paths = []
        for path in value["paths"].values():
            path = {
                "prefix": [p["prefix"] for p in path["prompts"]],
                "targets": [p["target"] for p in path["prompts"]],
            }
            paths.append(path)
        base_sequence = [
            {"role":"system", "content":value["system_message"]},
            {"role":"user", "content":value["user_message"]},
            {"role":"assistant", "content":value["reasoning"]},
        ]
        dist, scores, r_scores = model.evaluate(base_sequence, paths)
        result.update(dist)
        tracking[key] = {
            "scores":   scores.tolist(),
            "r_scores": r_scores,
        }
    output_file = os.path.join(config["datasets"]["score_output"], f"{model_name}.json")
    with open(output_file, "w") as f:
        json.dump(tracking, f, indent = 4)
    print()
    print({str(k): str(v) for k, v in result.items()})

def parse_arguments():
    with open("templates/help_string.txt", "r") as f:
        help_string = f.read().rstrip()
    model, dataset_name, n, partition = None, None, None, None
    all_models = {k for k in config["SLM"].keys() if isinstance(config["SLM"][k], dict)}
    all_datasets = set(config["datasets"]["all"])
    def fail(error = True):
        print(help_string)
        if error:raise ValueError(f"Invalid input. model = {all_models}, dataset = {all_datasets}")
    argumentList = sys.argv[1:]
    long_options, options = ["help", "model=", "dataset=", "partition=", "n="], "hm:d:p:n:"
    arguments, values = getopt.getopt(argumentList, options, long_options)
    for currentArgument, currentValue in arguments:
        match currentArgument:
            case "-h" | "--help": fail(False); return
            case "-m" | "--model":
                if currentValue in all_models: model = currentValue
                else: fail()
            case "-d" | "--dataset":
                if currentValue in all_datasets: dataset_name = currentValue
                else: fail()
            case "-p" | "--partition":
                assert dataset_name == "aime"
                try:
                    assert (0 < int(currentValue) <= 4)
                    partition = int(currentValue)
                except ValueError: fail()
            case "-n" | "--n":
                try: n = int(currentValue)
                except ValueError: fail()
            case _: fail()
    if model is None or dataset_name is None: fail()
    dataset_path = config["datasets"][dataset_name]["files"]["paths"]
    if partition is not None:
        assert dataset_name == "aime"
        dataset_path = dataset_path.replace("aime.json", f"aime_{partition}.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    if n is not None and n < len(dataset) and not config["SLM"][model]["cache"]:
        subsamp = set(np.random.choice(len(dataset), n, replace = False).tolist())
        samples = [(key, value) for i, (key, value) in enumerate(dataset.items()) if i in subsamp]
    else: samples = [(key, value) for key, value in dataset.items()]
    n = len(samples)
    print(f"Running analysis for...\nModel: {model}, Dataset: {dataset_name} with {n} samples (partition {partition})")
    generate_probability(model, samples)

if __name__ == "__main__":
    parse_arguments()
