#!/bin/python3.13
import os, toml, re, json, sys
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from Prompts import Prompter
from multiprocessing import Pool
from openai import OpenAI, OpenAIError
import time

config = toml.load("config.toml")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

name2openrouter = {
    "llama33_70B" :         "meta-llama/llama-3.3-70b-instruct",
    "deepseek_r1_70B" :     "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek_r1_32B" :     "deepseek/deepseek-r1-distill-qwen-32b",
    "gemma3_27B" :          "google/gemma-3-27b-it",
    "gemma3_12B" :          "google/gemma-3-12b-it",
    "qwen3_32B" :           "qwen/qwen3-32b",
}

def _call_with_retry(fn, max_retries=5, base_delay=1, **kwargs):
    for attempt in range(max_retries):
        try:
            resp = fn(**kwargs)
            return resp.choices[0].message.content
        except (OpenAIError, AttributeError) as exc:
            if attempt == max_retries - 1: raise
            sleep_for = base_delay * (2 ** attempt)
            time.sleep(sleep_for)

def get_response(sample):
    result = deepcopy(sample)
    result["model"] = sample["model"]
    response = _call_with_retry(
        client.chat.completions.create,
        model      = sample["model"],
        messages   = sample["prompt"],
        max_tokens = config["SLM"]["max_new"],
    )
    result["output"] = response
    return result

def get_cot_baseline_api(dataset, model_name):
    with open(config["datasets"][dataset]["files"]["test"], 'r') as f:
        test = json.load(f)
    prompter = Prompter(dataset, "Let's think step by step.\n", "prompt_template")
    prompts = []
    for key, value in test.items():
        inp = deepcopy(value)
        inp["unique_id"] = key
        inp["model"]     = name2openrouter[model_name]
        inp["prompt"]    = [
            {"role": "system", "content": prompter.get_system_message()},
            {"role": "user", "content": prompter.get_user_message(query = value["question"])},
        ]
        prompts.append(inp)
    with Pool(64) as p:
        all_outputs = list(
            tqdm(p.imap(get_response, prompts), total=len(prompts))
        )
    all_outputs = {i["unique_id"]:i for i in all_outputs}
    method = f"{model_name}_{dataset}_cot"
    output_file = os.path.join(config["datasets"]["cot_output"], f"{method}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=4)

if __name__ == "__main__":
    model = "gemma3_12B"
    get_cot_baseline_api("gsm8k", model)
    get_cot_baseline_api("math500", model)
    get_cot_baseline_api("aime", model)
