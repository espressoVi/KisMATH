#!/bin/python3.13
import os, toml, re, json
from tqdm import tqdm
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from Prompts import Prompter
from tqdm import tqdm

config = toml.load("config.toml")


class Paths:
    allowed_datasets = config["datasets"]["all"]
    MAXIMUM_PATHS = 200
    def __init__(self, dataset):
        assert dataset in self.allowed_datasets
        self.max_paths = config["datasets"][dataset]["max_paths"]
        self.prompter = Prompter(dataset, "", "prompt_template")
        graph_file = config["datasets"][dataset]["files"]["graph_output"]
        with open(graph_file, "r") as f:
            self.dataset = json.load(f)
        result = self.create_paths()
        with open(config["datasets"][dataset]["files"]["paths"], "w") as f:
            json.dump(result, f, indent = 4)

    def create_paths(self):
        result = {}
        for key, value in tqdm(self.dataset.items()):
            G = json_graph.adjacency_graph(value["graph"])
            paths = self._get_paths(G)
            ass_messages = self._path_to_completion(paths, value["reasoning"])
            result[key] = {
                "unique_id": value["unique_id"],
                "ground_truth": value["ground_truth"],
                "question": value["question"],
                "reasoning": value["reasoning"],
                "model_answer": value["model_answer"],
                "system_message": self.prompter.get_system_message(),
                "user_message": self.prompter.get_user_message(value["question"]),
                "paths": ass_messages,
            }
        return result

    def _path_to_completion(self, paths, reasoning):
        result = {}
        for i, path in enumerate(paths):
            res = {"path": "".join(path), "prompts":[]}
            for l, r in map(lambda x:eval(x), path):
                res["prompts"].append({
                    "prefix": reasoning[:l],
                    "target": reasoning[l:r].rstrip(),
                })
            result[i] = res
        return result

    def _get_paths(self, G):
        result = []
        q_nodes = [u for u in G.nodes if G.nodes[u]["label"][-1] == "question"]
        for qn in q_nodes:
            for cnt, path in enumerate(nx.all_simple_paths(G, qn, "answer")):
                if cnt > self.MAXIMUM_PATHS: break
                result.append(path)
        result = sorted(result, key = lambda x:len(x))[-self.max_paths:]
        result = [r[1:-1] for r in result]
        result = [r for r in result if len(r) > 0]
        return result
    
def main():
    #Paths("gsm8k")
    #Paths("math500")
    Paths("aime")

if __name__ == "__main__":
    main()
