#!/bin/python3.13
import os, re, toml, json
from datasets import load_dataset
import numpy as np

config = toml.load("config.toml")

class Sampler:
    """ Create few-shot samples and test sets for various datasets."""
    few_shot_num = config['datasets']['few_shot_num']
    allowed = set(config['datasets']['all'])
    def __init__(self, name):
        assert name in self.allowed
        self.dataset_name = name
        self.test_num = config['datasets'][name]['test_num']

    def create_datasets(self):
        raise NotImplementedError
    def process(self, sample):
        raise NotImplementedError

    def sample(self):
        if (
            os.path.exists(config["datasets"][self.dataset_name]["files"]["few_shots"]) 
            or os.path.exists(config["datasets"][self.dataset_name]["files"]["test"])
        ): return
        self.base_dataset = load_dataset(*config['datasets'][self.dataset_name]['name'])
        self.create_datasets()
        with open(config["datasets"][self.dataset_name]["files"]["few_shots"], "w") as f:
            json.dump(self.few_shot, f, indent = 4)
        with open(config["datasets"][self.dataset_name]["files"]["test"], "w") as f:
            json.dump(self.test, f, indent = 4)

class GSM8K(Sampler):
    def __init__(self):
        super().__init__("gsm8k")
        self.sample()

    def create_datasets(self):
        idxs = np.random.choice(
                len(self.base_dataset["train"]),
                self.few_shot_num,
                replace = False,
        ).tolist()
        self.few_shot = {i:self.process(self.base_dataset["train"][i]) for i in idxs}
        idxs = np.random.choice(
                len(self.base_dataset["test"]),
                self.test_num,
                replace = False,
        ).tolist()
        self.test = {i:self.process(self.base_dataset["test"][i]) for i in idxs}

    def process(self, sample):
        ans = sample["answer"].split("\n")
        ans = [re.sub(r"<<[^>]*>>", "", line).strip() for line in ans]
        res = re.sub(r"#### ", "", ans[-1]).strip()
        result = {
            "question": sample["question"],
            "original_answer": sample["answer"],
            "original_reasoning": ans[:-1],
            "ground_truth": res,
        }
        return result

class MATH500(Sampler):
    def __init__(self):
        super().__init__("math500")
        self.sample()

    def create_datasets(self):
        dataset = [i for i in self.base_dataset["test"] \
                        if "[asy]" not in i["problem"] \
                        and "[asy]" not in i["solution"] \
                        and "diagram" not in i["problem"].lower() \
                        and i["subject"] != "Geometry"
        ]
        hardest = np.random.permutation([j for j,i in enumerate(dataset) if i["level"] == 5]).tolist()
        idxs, subjects = [], set()
        for i in hardest:
            if dataset[i]["subject"] in subjects:
                continue
            subjects.add(dataset[i]["subject"])
            idxs.append(i)
        fidxs = set(np.random.choice(idxs, self.few_shot_num, replace = False,).tolist())
        self.few_shot = {i:self.process(dataset[i]) for i in fidxs}
        idxs = set(list(range(len(dataset)))) - fidxs
        self.test = {i:self.process(dataset[i]) for i in idxs}

    def process(self, sample):
        result = {
            "question": sample["problem"],
            "original_answer": sample["solution"],
            "ground_truth": sample["answer"],
            "dataset_id": sample["unique_id"],
            "subject": sample["subject"],
            "level": sample["level"],
        }
        return result

class AIME(Sampler):
    def __init__(self):
        super().__init__("aime")
        self.sample()

    def create_datasets(self):
        dataset = []
        gg_regex = re.compile(r".*(diagram|trapez|tetrahedron|quadrilateral|square|triangle|triang|pentagon|rectangle|rhombus|align|\[asy\]|hexagon|octagon).*")
        for ds in self.base_dataset["train"]:
            if gg_regex.match(ds["Question"].lower()): continue
            try: int(ds["Answer"])
            except: continue
            dataset.append(ds)
        fidxs = set(np.random.choice(len(dataset), self.few_shot_num, replace = False).tolist())
        self.few_shot = {i:self.process(dataset[i]) for i in fidxs}
        idxs = list(set(list(range(len(dataset)))) - fidxs)
        fidxs = set(np.random.choice(idxs, self.test_num, replace = False,).tolist())
        self.test = {i:self.process(dataset[i]) for i in fidxs}

    def process(self, sample):
        question = re.sub(r"[\^_]\{\}", "", sample["Question"])
        result = {
            "question": question,
            "ground_truth": sample["Answer"],
            "dataset_id": sample["ID"],
        }
        return result

def main():
    AIME()

if __name__ == "__main__":
    main()

