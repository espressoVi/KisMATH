#!/bin/python3
import re, toml, json, os
import torch
import torch.nn.functional as F
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
        total_length, offset = input.shape[-1], input.shape[-1] - reasoning.shape[-1]
        masks = {k:self.get_attention_mask(path, total_length, offset) for k, path in paths.items()}
        self.load_model()
        torch.cuda.empty_cache()
        base_distr, base_entropy = self.get_distribution(input, torch.ones_like(input))
        if masks:
            max_path = (max(masks.values(), key = lambda x:torch.sum(x==0))==0).sum().item()
            idx = torch.tensor(np.random.choice(input.shape[-1], max_path, replace = False), dtype = int)
            rand_mask = torch.ones_like(input)
            rand_mask[:, idx] = 0
            rand = self.get_distribution(input, rand_mask, base_distr)
        else: rand = None
        result = {}
        for key, mask in masks.items():
            rr = self.get_distribution(input, mask, base_distr)
            result[key] = rr
        return base_entropy, rand, result

    def graph_evaluate(self, prompt, language, maths, suffix):
        input = self.tokenizer.apply_chat_template(
            prompt,
            tokenize = False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        input = self.tokenizer.encode(input, return_tensors = "pt")[:, :-self.idx]
        reasoning = self.tokenizer.encode(prompt[-1]["content"], return_tensors = "pt")[:, self.bidx:]
        trunc = input[:, :-reasoning.shape[-1]]
        pre_mask = [1]*trunc.shape[-1]
        math_mask = []
        for lang, math in zip(language[:-1], maths):
            l = self.tokenizer.encode(lang, return_tensors = "pt")[:, self.bidx:]
            m = self.tokenizer.encode(math, return_tensors = "pt")[:, self.bidx:]
            math_mask.extend([1]*l.shape[-1])
            math_mask.extend([0]*m.shape[-1])
            trunc = torch.cat((trunc, l, m), 1)
        l = self.tokenizer.encode(language[-1], return_tensors = "pt")[:, self.bidx:]
        trunc = torch.cat((trunc, l), 1)
        s = self.tokenizer.encode(suffix, return_tensors = "pt")[:, self.bidx:].shape[-1]
        math_mask.extend([1]*(l.shape[-1] - s))
        math_mask = torch.tensor(math_mask).unsqueeze(0)
        pre_mask = torch.tensor(pre_mask).unsqueeze(0)
        post_mask = torch.ones((1, s), dtype = int)
        graph_mask = torch.cat((pre_mask, math_mask, post_mask), 1)     # Math masked
        graph_C_mask = torch.cat((pre_mask, 1-math_mask, post_mask), 1) # Language masked
        norm_mask = torch.ones_like(trunc)                              # Nothing masked
        self.load_model()
        torch.cuda.empty_cache()
        nscore, nent = self.get_distribution(trunc, norm_mask)
        nans = self.tokenizer.decode([torch.argmax(nscore)])
        mscore, ment = self.get_distribution(trunc, graph_C_mask)
        mans = self.tokenizer.decode([torch.argmax(mscore)])
        lscore, lent = self.get_distribution(trunc, graph_mask)
        lans = self.tokenizer.decode([torch.argmax(lscore)])
        return {
            "orig_entropy": nent,
            "math_masked_entropy": lent,
            "lang_masked_entropy": ment,
            "orig_answer": nans,
            "math_masked_ans": lans,
            "lang_masked_ans": mans,
        }

    def get_attention_mask(self, path, total_length, offset):
        mask = torch.ones((1, total_length), dtype=int)
        for p, t in zip(path["prefix"], path["targets"]):
            pref = self.tokenizer.encode(p, return_tensors = "pt")[:, self.bidx:].squeeze(0)
            targ = self.tokenizer.encode(t, return_tensors = "pt")[:, self.bidx:].squeeze(0)
            mask[0, offset+len(pref): offset+len(pref) + len(targ)] = 0
        return mask

    def get_distribution(self, input, mask, comp_scores = None):
        with torch.inference_mode():
            results = self.model.generate(
                input_ids = input.to("cuda"),
                attention_mask = mask.to("cuda"),
                generation_config = self.generation_config,
            )
        score = F.log_softmax(results.scores[0], dim = -1).squeeze().detach().cpu()
        m = torch.isfinite(score)
        entropy = -torch.sum(score[m]*score[m].exp()).item()
        if comp_scores is None:
            return score, entropy
        m = torch.isfinite(score) & torch.isfinite(comp_scores)
        kl = F.kl_div(score, comp_scores, log_target = True, reduction = "batchmean").item()
        return {"KL": max(0, kl), "H":entropy}
