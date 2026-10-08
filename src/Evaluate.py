#!/bin/python3.13
from openai import OpenAI
import os, re, toml, json
from copy import deepcopy
from sympy.parsing.latex import parse_latex
import sympy

config = toml.load("config.toml")
client = OpenAI()


class EvaluateBase:
    allowed = set(config["datasets"]["all"])
    def __init__(self, dataset):
        assert dataset in self.allowed
        self.dataset = dataset

    def __call__(self, output_file):
        with open(output_file, "r") as f:
            data = json.load(f)
        is_processed = all(["correct" in i for i in data.values()])
        if is_processed:
            return sum([1 for i in data.values() if i["correct"]])/(len(data))
        result = deepcopy(data)
        counter = 0
        for key, val in data.items():
            model_output = val["output"]
            model_output = self._strip_context(model_output, val["prompt"][-1]["content"])
            model_output = re.sub(r"<\|[^\|]+\|>", "", model_output)
            prediction = self._get_prediction(model_output)
            prediction = self.postprocess(prediction)
            ground_truth = self.postprocess(val["ground_truth"])
            if prediction is not None and prediction == ground_truth:
                counter += 1
                result[key]["correct"] = True
            elif self.symbolic_judge(val["question"], ground_truth, prediction):
                counter += 1
                result[key]["correct"] = True
            elif self.model_judge(val["question"], ground_truth, prediction):
                counter += 1
                result[key]["correct"] = True
            else:
                result[key]["correct"] = False
        with open(output_file, "w") as f:
            json.dump(result, f, indent = 4)
        return counter/(len(data))

    def _get_prediction(self, model_output):
        for line in reversed(model_output.split("\n")):
            if match := re.search(self.answer_regex, line):
                return match.group(1) if match.group(1) is not None else match.group(2)
        return None

    def _strip_context(self, output, context):
        try:
            return output[output.rindex(context):]
        except ValueError:
            return output

    def postprocess(self, answer):
        raise NotImplementedError

    def model_judge(self, question, answer1, answer2):
        raise NotImplementedError

class EvaluateGSM8K(EvaluateBase):
    def __init__(self):
        super().__init__("gsm8k")
        self.answer_regex = r"^.*\(([^)]+)\).*$|^.*\\boxed\{(.*)\}.*$"

    def symbolic_judge(self, question, answer1, answer2):
        return False

    def model_judge(self, question, answer1, answer2):
        return False

    def postprocess(self, answer):
        if answer is None:return None
        answer = answer.strip()
        answer = answer.replace(r"boxed", "")
        answer = re.sub(r"[^0-9]", "", answer)
        for match in re.finditer(r"\d+(,)\d{3}", answer):
            answer = answer.replace(",", "")
        return answer.strip().replace(",","")

class EvaluateMATH(EvaluateBase):
    def __init__(self):
        super().__init__("math500")
        self.answer_regex = r"^.*\\boxed\{(.*)\}.*$"
        with open(config["datasets"]["evaluation_prompt"], "r") as f:
            self.system_message = f.read().strip()

    def symbolic_judge(self, question, answer1, answer2):
        if answer1 is None or answer2 is None:
            return False
        try:
            f1 = parse_latex(answer1)
            f2 = parse_latex(answer2)
        except (sympy.parsing.latex.errors.LaTeXParsingError, ValueError) as e:
            return False
        try:
            ret = (sympy.simplify(f1-f2) == 0)
        except TypeError:
            return False
        return ret

    def model_judge(self, question, answer1, answer2):
        return False
        usr_message = f"Question: {question}\n\nAnswer 1: {answer1}\nAnswer 2: {answer2}"
        response = client.responses.create(
            model = "gpt-4.1",
            input = [
                {"role":"system", "content":self.system_message},
                {"role":"system", "content":usr_message},
            ],
        )
        return response.output[0].content[0].text.lower().strip() == "yes"

    def postprocess(self, answer):
        if answer is None:return None
        answer = answer.strip()
        answer = answer.replace("dfrac", "frac")
        answer = answer.replace("textbf", "text")
        answer = re.sub(r"\\[!;,$%]", "", answer)
        answer = re.sub(r"\s+", "", answer)
        if match := re.search(r"0\.\d+", answer):
            answer = answer.replace("0.",".")
        for match in re.finditer(r"\{\d\}", answer):
            answer = answer.replace(match.group(0), match.group(0)[1:-1])
        for match in re.finditer(r"\d+(,)\d{3}", answer):
            answer = answer.replace(",", "")
        return answer


if __name__ == "__main__":
    #print(EvaluateGSM8K()("./outputs/llm_output/gsm8k_o3_reasoning_output.json"))
    #print(EvaluateMATH()("./outputs/llm_output/math500_o3_reasoning_output.json"))
    #print(EvaluateMATH()("./outputs/llm_output/aime_o3_reasoning_output.json"))
    directory = "./outputs/baselines/"
    all_files = sorted(os.listdir(directory))
    for file in all_files:
        file = os.path.join(directory, file)
        if "gsm8k" in file:
            res = EvaluateGSM8K()(file)
        else:
            res = EvaluateMATH()(file)
        ff = file.split("/")[-1].split(".")[0]
        print(f"{ff} : Accuracy: {round(100*res, 1)}")
        print("-"*80)
