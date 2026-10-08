#!/bin/python3
import re, toml, json, os
import torch
import accelerate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

config = toml.load("config.toml")


class SLM:
    """ LLM class. Implements inference """
    def __init__(self, name):
        self.name = name
        self.llm_path = config['SLM'][name]['path']
        self.loaded = False
        self.idx = config["SLM"][name]["offset"]
        self.initialize()

    def initialize(self):
        """ Inititalize tokenizer and generation config. """
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code = True)
        self.tokenizer.unk_token = "<notrequired>" 
        self.tokenizer.sep_token = "<notrequired>"
        self.tokenizer.pad_token = "<notrequired>"
        self.tokenizer.cls_token = "<notrequired>"
        self.tokenizer.mask_token = "<notrequired>"
        self.generation_config = transformers.GenerationConfig(
            do_sample = True,
            max_new_tokens = config['SLM']['max_new'],
            #temperature = None,
            #top_k = None,
            #top_p = None,
            num_return_sequences = 1,
            pad_token_id = self.tokenizer.eos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            return_dict_in_generate = True,
            output_scores = True,
        )

    def load_model(self):
        if self.loaded: return
        self.model = AutoModelForCausalLM.from_pretrained(
                    self.llm_path,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    trust_remote_code = True,
                )
        self.model.eval()
        self.loaded = True

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Run inference and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": You are a helpful AI assistant...},
              {"role": "user", "content": prompt},
        ]
        -------------------
        Args:
            list[dict] : As described above.
        Returns:
            list[str]: Completion by SLM.
        """
        self.load_model()
        torch.cuda.empty_cache()
        prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize = False,
            add_generation_prompt=False,
            enable_thinking=False
        ).rstrip()
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")[:, :-self.idx]
        input_token_length = len(inputs[0])
        with torch.inference_mode():
            results = self.model.generate(
                input_ids = inputs.to("cuda"),
                attention_mask = torch.ones_like(inputs).to("cuda"),
                generation_config = self.generation_config,
            )
        output = self.tokenizer.decode(
            results.sequences[0][input_token_length:],
            skip_special_tokens=True
        ) 
        return output
