#!/bin/python3.13
import os, toml, re, json
from jinja2 import Template
from copy import deepcopy
from tqdm import tqdm

config = toml.load("config.toml")


class Prompter:
    usr_template = Template("Question: {{ query }}")
    def __init__(self, dataset, ass_template, sys_template):
        self.ass_template = Template(ass_template)
        with open(config["datasets"][dataset]["files"][sys_template], "r") as f:
            self.sys_template = Template(f.read().strip())
        with open(config["datasets"][dataset]["files"]["few_shots"], 'r') as f:
            examples = json.load(f)
        self.system_message = self.sys_template.render(examples = examples).rstrip()

    def get_system_message(self):
        return self.system_message

    def get_user_message(self, query):
        usr_message = self.usr_template.render(query = query)
        return usr_message

    def get_assistant_message(self, **kwargs):
        ass_message = self.ass_template.render(**kwargs)
        return ass_message

    def remove_ground_truth(self, reasoning, ground_truth):
        try:
           idx = reasoning[-1].rindex(ground_truth)
        except ValueError:
            return reasoning
        reasoning[-1] = reasoning[-1][:idx]
        return reasoning

