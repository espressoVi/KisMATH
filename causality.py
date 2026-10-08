#!/bin/python3.13
import os, toml, re, json, sys, getopt
from jinja2 import Template
from copy import deepcopy
from tqdm import tqdm
from src.PathCausal import PathLM
from collections import Counter
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from src.Prompts import Prompter

config = toml.load("config.toml")

def check_graphs(dataset_name, model_name):
    with open(config["datasets"][dataset_name]["files"]["graph_output"], 'r') as f:
        samples = json.load(f)
    suffix = "\nAnswer: (" if dataset_name == "gsm8k" else "\nAnswer: \\boxed{"
    prompter = Prompter(dataset_name, "", "prompt_template")
    model = PathLM(model_name)
    result = {}
    for key, value in tqdm(samples.items()):
        G = json_graph.adjacency_graph(value["graph"])
        r_nodes = [eval(n) for n, m in nx.get_node_attributes(G, 'label').items() if m[1] == "reasoning"]
        r_nodes = sorted(r_nodes, key = lambda x:x[0])
        rollout = value["reasoning"] + suffix
        base_sequence = [
            {"role": "system",    "content": prompter.get_system_message()},
            {"role": "user",      "content": value["question"]},
            {"role": "assistant", "content": rollout},
        ]
        r, l_segment, m_segment = 0, [], []     # Language segment, math segment
        for start, end in r_nodes:
            l_segment.append(rollout[r:start])
            m_segment.append(rollout[start:end])
            r = end 
        l_segment.append(rollout[r:])
        res = model.graph_evaluate(base_sequence, l_segment, m_segment, suffix)
        result[key] = res
    method = f"{model_name}_{dataset_name}_graph"
    output_file = os.path.join(config["datasets"]["path_output"], f"{method}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent = 4)

def check_paths(dataset_name, model_name):
    with open(config["datasets"][dataset_name]["files"]["paths"], 'r') as f:
        samples = json.load(f)
    suffix = "\nAnswer: (" if dataset_name == "gsm8k" else "\nAnswer: \\boxed{"
    model = PathLM(model_name)
    result = {}
    for key, value in tqdm(samples.items()):
        paths = {}
        for pk, path in value["paths"].items():
            one_path = {
                "prefix": [p["prefix"] for p in path["prompts"]],
                "targets": [p["target"] for p in path["prompts"]],
            }
            assert one_path["prefix"][-1] in value["reasoning"]
            paths[pk] = one_path
        base_sequence = [
            {"role":"system", "content":value["system_message"]},
            {"role":"user", "content":value["user_message"]},
            {"role":"assistant", "content":value["reasoning"] + suffix},
        ]
        orig_entropy, random_stat, path_stat = model.evaluate(base_sequence, paths)
        result[key] = {
            "original_entropy": orig_entropy,
            "random_path":      random_stat,
            "stats":            path_stat,
        }
    method = f"{model_name}_{dataset_name}_pp"
    output_file = os.path.join(config["datasets"]["path_output"], f"{method}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent = 4)

def parse_arguments():
    help_string = "python generate_baselines.py --model [model_name] --dataset [dataset_name]\n"
    model, dataset_name, graph = None, None, False
    all_models = {k for k in config["SLM"].keys() if isinstance(config["SLM"][k], dict)}
    all_datasets = set(config["datasets"]["all"])
    argumentList = sys.argv[1:]
    long_options, options = ["help", "model=", "dataset=", "graph", "path"], "hm:d:"
    arguments, values = getopt.getopt(argumentList, options, long_options)
    def fail(error = True):
        print(help_string)
        if error:raise ValueError(f"Invalid input. model = {all_models}, dataset = {all_datasets}")
    for currentArgument, currentValue in arguments:
        match currentArgument:
            case "-h" | "--help": fail(False); return
            case "--graph": graph = True
            case "--path": graph = False
            case "-m" | "--model":
                if currentValue in all_models: model = currentValue
                else: fail()
            case "-d" | "--dataset":
                if currentValue in all_datasets: dataset_name = currentValue
                else: fail()
    if model is None or dataset_name is None: fail()
    if graph is False:
        print(f"Checking if paths affect answer...\nModel: {model}, Dataset: {dataset_name}\n", file=sys.stderr)
        check_paths(dataset_name, model)
    else:
        print(f"Checking how full graphs affect answer...\nModel: {model}, Dataset: {dataset_name}\n", file=sys.stderr)
        check_graphs(dataset_name, model)

if __name__ == "__main__":
    parse_arguments()
