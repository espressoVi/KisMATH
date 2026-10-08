#!/bin/python3
import re, toml, json, os
import torch
import accelerate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from src.Generator import SLM

config = toml.load("config.toml")


class PathLM(SLM):
    """ LLM class. Implements calculation of path probabilities. """
    def __init__(self, name):
        super().__init__(name)
        self.idx = config["SLM"][name]["offset"]
        self.bidx = config["SLM"][name]["is_bos"]
        self.cache = config["SLM"][name]["cache"]
        self.generation_config.max_new_tokens = 1
        self.generation_config.temperature = config['SLM']['temperature']
        self.generation_config.top_k = None
        self.generation_config.top_p = None
        self.K = 10

    def evaluate(self, base_sequence, paths):
        """ Calculate the probability of a path, and K random paths."""
        input = self.tokenizer.apply_chat_template(
            base_sequence,
            tokenize = False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        input = self.tokenizer.encode(input, return_tensors = "pt")[:, :-self.idx]
        reasoning = self.tokenizer.encode(base_sequence[-1]["content"], return_tensors = "pt")[:, self.bidx:]
        scores = self.get_scores(input, reasoning) 
        input = input.detach().cpu().numpy().squeeze()
        reasoning = reasoning.detach().cpu().numpy().squeeze()
        input = input[ :input.shape[0]-reasoning.shape[0]]
        answer, r_scores = [], []
        for path in paths:
            running_length, running_score = 0, 0
            for prefix, target in zip(path["prefix"], path["targets"]):
                pr = self.tokenizer.encode(prefix, return_tensors = "np").squeeze(0).shape[0]
                tr = self.tokenizer.encode(target, return_tensors = "np").squeeze(0).shape[0]
                running_score += np.sum(scores[pr:pr+tr])
                running_length += tr
            random_path = []
            r_scores.append(float(running_score))
            for i in range(self.K):
                random_path.append(np.random.choice(scores, running_length, replace = False).sum())
            random_path = np.array(random_path) - running_score
            rank = np.sum(np.where(random_path <= 0, 1, 0))
            answer.append(int(rank))
        return answer, scores, r_scores

    def get_scores(self, input, reasoning):
        assert torch.all(torch.where(input[:, -reasoning.shape[-1]:] == reasoning, True, False))
        past_key_values = None
        scores = []
        inp = input.detach().clone()[:, :input.shape[-1]-reasoning.shape[-1]]
        target = reasoning.detach().clone()
        self.load_model()
        torch.cuda.empty_cache()
        for i in range(target.shape[-1]):
            with torch.inference_mode():
                results = self.model.generate(
                    input_ids = inp.to("cuda"),
                    attention_mask = torch.ones_like(inp).to("cuda"),
                    generation_config = self.generation_config,
                    past_key_values = past_key_values,
            )
            past_key_values = results.past_key_values if self.cache else None
            score = torch.nn.functional.log_softmax(results.scores[0], dim = -1).squeeze().detach().cpu().numpy()
            scores.append(score[target[0, i]])                                  # Score at different positions.
            inp = torch.cat((input, target[:, :i+1]), dim = -1)                 # Forced decoding.
        return np.array(scores)
