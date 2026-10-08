#!/bin/python3.13
import os, toml, re, json, sys, getopt
from src.Generator import SLM
from jinja2 import Template
from copy import deepcopy
from tqdm import tqdm
from src.Prompts import Prompter

config = toml.load("config.toml")


def get_cot_baseline(dataset, model_name):
    with open(config["datasets"][dataset]["files"]["test"], 'r') as f:
        test = json.load(f)
    prompter = Prompter(dataset, "Let's think step by step.\n", "prompt_template")
    prompts = {}
    for key, value in test.items():
        inp = deepcopy(value)
        inp["prompt"] = [
            {"role": "system", "content": prompter.get_system_message()},
            {"role": "user", "content": prompter.get_user_message(query = value["question"])},
            {"role": "assistant", "content": prompter.get_assistant_message()},
        ]
        prompts[key] = inp
    model = SLM(model_name)
    method = f"{model_name}_{dataset}_cot"
    results = model_inference(model, prompts)
    output_file = os.path.join(config["datasets"]["cot_output"], f"{method}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

def model_inference(model, prompts):
    answer = {}
    for key, value in tqdm(prompts.items()):
        res = deepcopy(value)
        res["output"] = model(value["prompt"])
        answer[key] = res
    return answer

def parse_arguments():
    help_string = "python generate_baselines.py --model [model_name] --dataset [dataset_name]\n"
    model, dataset_name = None, None
    all_models = {k for k in config["SLM"].keys() if isinstance(config["SLM"][k], dict)}
    all_datasets = set(config["datasets"]["all"])
    argumentList = sys.argv[1:]
    long_options, options = ["help", "model=", "dataset="], "hm:d:"
    arguments, values = getopt.getopt(argumentList, options, long_options)
    def fail(error = True):
        print(help_string)
        if error:raise ValueError(f"Invalid input. model = {all_models}, dataset = {all_datasets}")
    for currentArgument, currentValue in arguments:
        match currentArgument:
            case "-h" | "--help": fail(False); return
            case "-m" | "--model":
                if currentValue in all_models: model = currentValue
                else:fail()
            case "-d" | "--dataset":
                if currentValue in all_datasets: dataset_name = currentValue
                else:fail()
    if model is None or dataset_name is None:fail()
    print(f"Running Chain-of-thought baseline...\nModel: {model}, Dataset: {dataset_name}")
    get_cot_baseline(dataset_name, model)

def main():
    parse_arguments()

if __name__ == "__main__":
    main()
