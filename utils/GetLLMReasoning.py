import os, re, toml, json, sys
from jinja2 import Template
from copy import deepcopy
from openai import OpenAI, OpenAIError
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from Prompts import Prompter
import time

config = toml.load("config.toml")
client = OpenAI()

def _call_with_retry(fn, max_retries=5, base_delay=1, **kwargs):
    for attempt in range(max_retries):
        try:
            resp = fn(**kwargs)
            return resp.output[1].content[0].text
        except (OpenAIError, AttributeError) as exc:
            if attempt == max_retries - 1: raise
            sleep_for = base_delay * (2 ** attempt)
            time.sleep(sleep_for)

def get_openai_response(sample):
    result = deepcopy(sample)
    model = config["LLM"]["o3"]["name"]
    result["model"] = model
    response = _call_with_retry(
        client.responses.create,
        model=model,
        reasoning = {"effort": "medium"},
        input = result["prompt"],
    )
    result["output"] = response
    return result

def extract_openai(dset):
    with open(config["datasets"][dset]["files"]["test"], 'r') as f:
        test = json.load(f)
    prompter = Prompter(dset, "", "prompt_template")
    prompts = []
    for key, value in test.items():
        inp = deepcopy(value)
        inp["unique_id"] = key
        inp["prompt"] = [
            {"role": "system", "content": prompter.get_system_message()},
            {"role": "user", "content": prompter.get_user_message(query = value["question"])},
        ]
        prompts.append(inp)
    with Pool(64) as p:
        all_outputs = list(
            tqdm(p.imap(get_openai_response, prompts), total=len(prompts))
        )
    all_outputs = {i["unique_id"]:i for i in all_outputs}
    with open(config["datasets"][dset]["files"]["llm_output"], "w") as f:
        json.dump(all_outputs, f, indent=4)


if __name__ == "__main__":
    extract_openai("aime")
