#!/bin/python3.13
import os, toml, re, json
from jinja2 import Template
from copy import deepcopy
from openai import OpenAI
from tqdm import tqdm
import networkx as nx
from networkx.readwrite import json_graph
import sympy
from sympy.parsing.latex import parse_latex
from itertools import product
import matplotlib.pyplot as plt
from TextNumbers import TextNumbers
from Display import Draw

plt.rcParams['text.usetex'] = False

config = toml.load("config.toml")


class Tracer:
    answer_regex = r"^.*\\boxed\{(.*)\}.*$"
    equivs = re.compile(r"=|\\(Long)?[rR]ightarrow|\\[lL]eftarrow|\\[lL]eftrightarrow|\\implies|,|\\ne")
    inequs = re.compile(r"<|>|\\le[q]?|\\ge[q]?")
    textnumber = TextNumbers()
    allowed = set(config['datasets']['all'])

    def __init__(self, dataset):
        self.dataset_name = dataset
        output_file = config["datasets"][dataset]["files"]["llm_output"]
        with open(output_file, "r") as f:
            self.model_output = json.load(f)

    def __call__(self):
        result = {}
        samples = [(key, sample) for key, sample in self.model_output.items()]
        for key, sample in tqdm(samples):
            if not sample["correct"]:continue
            model_output = sample["output"]
            model_output = self._strip_context(model_output, sample["prompt"][-1]["content"])
            answer, reasoning = self._preprocess(model_output)
            question = self._filter(sample["question"])
            graph = self.trace(answer, reasoning, question)
            result[key] = {
                "unique_id": sample["unique_id"],
                "ground_truth": sample["ground_truth"],
                "question" : question,
                "reasoning": reasoning,
                "model_answer": answer,
                "graph": json_graph.adjacency_data(graph),
            }
        print(len(result))
        with open(config["datasets"][self.dataset_name]["files"]["graph_output"], "w") as f:
            json.dump(result, f, indent = 4)

    def trace(self, answer, reasoning, question):
        self.matched, G = {}, nx.DiGraph()
        G.add_node("answer", label = (answer, "answer"))
        panswer = self._parse_formula(answer, (0, len(answer)))
        formulas = self._get_formulas(reasoning)
        qformulas = self._get_question_terms(question)
        self.min_context = len(qformulas)
        all_formulas = qformulas + formulas
        for form in all_formulas:
            form["parsed"] = self._parse_formula(form["text"], form["idx"])
        self._node_expansion(panswer, all_formulas, G, "answer")
        G = self._prune_graph(G)
        return G

    def _node_expansion(self, search_terms, context, graph, node):
        if len(context) < self.min_context or len(search_terms) == 0:
            return
        search_terms = sorted(search_terms, key = lambda x: len(x["text"]))
        for cnt, formula in enumerate(reversed(context)):
            for term, search_term in product(formula["parsed"], search_terms):
                if (
                    (not graph.has_edge(node, str(term["idx"])))
                    and self._match(term, search_term)
                ):
                    graph.add_node(str(term["idx"]), label = (term["text"], formula["type"]))
                    graph.add_edge(node, str(term["idx"]))
                    self._node_expansion(
                        formula["parsed"],
                        context[:len(context) - cnt -1],
                        graph,
                        str(term["idx"]),
                    )
        return

    def _match(self, t1, t2):
        if self._term_hash(t1)+self._term_hash(t2) in self.matched or self._term_hash(t2)+self._term_hash(t1) in self.matched:
            return self.matched.get(self._term_hash(t1)+self._term_hash(t2), False) or self.matched.get(self._term_hash(t1)+self._term_hash(t2), False)
        if t1["text"] == t2["text"] or t2["text"] in t1["text"]:
            return True
        if "parsed" in t1 and "parsed" in t2:
            l1 = [e for e in sympy.postorder_traversal(t1["parsed"])]
            l2 = [e for e in sympy.postorder_traversal(t2["parsed"])]
            for i, j in product(l1, l2):
                if type(i) != type(j):continue
                try:
                    ret = (sympy.simplify(i-j) == 0) or (sympy.simplify(i+j) == 0)
                except TypeError:
                    continue
                if ret:
                    self.matched[self._term_hash(t1) + self._term_hash(t2)] = ret
                    return ret
        self.matched[self._term_hash(t1) + self._term_hash(t2)] = False
        return False

    @staticmethod
    def _term_hash(t1):
        return t1["idx"] + (t1["text"],)

    def _parse_formula(self, formula, span):
        if len(formula) == 0:
            return []
        if (
                len(formula) == 1 or
                (m := re.match(r"^-?\d+\.?\d+$", formula)) or
                (m := re.match(r"^\w+$", formula))
        ):
            try:
                ff = parse_latex(formula)
            except (sympy.parsing.latex.errors.LaTeXParsingError, ValueError) as e:
                return [{"idx":span, "text":formula}]
            return [{"idx":span, "text":formula, "parsed":ff}]
        all_parts = list(self.equivs.finditer(formula)) + list(self.inequs.finditer(formula))
        for m in all_parts:
            if not self._is_scope_global(formula, m.span()): continue
            return (
                self._parse_formula(formula[:m.span()[0]], (span[0], span[0]+m.span()[0])) +
                self._parse_formula(formula[m.span()[1]:], (span[0]+m.span()[1], span[1]))
            )
        if m:= re.match(r"^(\d+[\./]?\d*)\\%\s*$", formula):
            return [{"idx":(span[0]+m.span(1)[0], span[0]+m.span(1)[1]),
                     "text":m.group(1), "parsed":parse_latex(f"{m.group(1)}/100")}]
        try:
            ff = parse_latex(formula)
        except (sympy.parsing.latex.errors.LaTeXParsingError, ValueError) as e:
            return [{"idx":span, "text":formula}]
        return [{"idx":span, "text":formula, "parsed":ff}]

    def _get_question_terms(self, question):
        result = []
        for match in re.finditer(r"(?:\$|\\\[|\\\])?\b(\d+\.\d+|\d+/\d+|(?<!:)\d+(?=(?:\.(?!\d))?(\W|$)))", question):
            result.append(
                {"idx": match.span(1), "text": match.group(1), "type":"question"}
            )
        for match in re.finditer(r"(\d+[\./]?\d*%)\s+", question):
            result.append({
                "idx": match.span(1),
                "text": match.group(1)[:-1]+"/100",
                "type":"question"
             })
        for match in re.finditer(r"(?<!\\)(?:\$((?:\\\$|[^$])*?)(?<!\\)\$|\\\[((?:\\\]|[^]])*?)(?<!\\)\\\])", question, re.DOTALL):
            result.append({
                    "idx": match.span(1) if match.group(1) is not None else match.span(2),
                    "text": match.group(1) if match.group(1) is not None else match.group(2),
                    "type": "question",
            })
        result += self.textnumber(question)
        res = []
        for i, a in enumerate(result):
            if not any(i != j and a["idx"][0] >= b["idx"][0] and a["idx"][1] <= b["idx"][1] for j, b in enumerate(result[:i])):
                res.append(a)
        return res

    def _get_formulas(self, line):
        result = []
        for match in re.finditer(r"(?<!\\)\$((?:\\\$|[^$])*)(?<!\\)\$", line):
            result.append({
                    "idx": match.span(1),
                    "text": match.group(1),
                    "type": "reasoning",
            })
        return result

    def _preprocess(self, output):
        answer = None
        for line in reversed(output.split("\n")):
            if match := re.search(self.answer_regex, line):
                answer, ans_line = match.group(1), line
                break
        if answer is None:
            raise ValueError("Couldn't find answer.")
        answer = self._filter(answer)
        reasoning = output[:output.rindex(ans_line)]
        reasoning = self._filter(reasoning)
        return answer, reasoning

    @staticmethod
    def _filter(text):
        text = re.sub(r"<\|[^\|]+\|>", "", text)
        text = re.sub(r"\\displaystyle|\\big[lr]|\\[lr]angle|\\mathbf|\^\\circ", " ", text)
        text = re.sub(r"\\left\(", "(", text)
        text = re.sub(r"\\right\)", ")", text)
        text = re.sub(r"\\left[^(a]|\\right[^)a]", "", text)
        text = re.sub(r"\\[;!,$>\s]", "", text)
        text = re.sub(r"\\[q]+uad", "", text)
        text = text.strip()
        return text

    def _prune_graph(self, G):
        r_nodes = [n for n, m in nx.get_node_attributes(G, 'label').items() if m[1] == "reasoning"]
        q_nodes = [n for n, m in nx.get_node_attributes(G, 'label').items() if m[1] == "question"]
        for s, k in product(r_nodes, r_nodes):
            if s == k:continue
            if eval(s)[0] >= eval(k)[0] and eval(s)[1] <= eval(k)[1] and s in G:
                G.add_edges_from([(k, edge[1]) if edge[0] == s else (edge[0], k) for edge in G.edges(s)])
                G.remove_node(s)
        r_nodes = [n for n, m in nx.get_node_attributes(G, 'label').items() if m[1] == "reasoning"]
        for s in r_nodes:
            if any([nx.has_path(G, s, t) for t in q_nodes]): continue
            G.remove_node(s)
        G = G.reverse()
        G.remove_edges_from([(u, u) for u in G.nodes if G.has_edge(u,u)])               # Accidental self-loops
        back = [(u, v) for u, v in G.edges if G.nodes[u]["label"][-1] == "question" \
                                            and G.nodes[v]["label"][-1] == "question"]
        G.remove_edges_from(back)                                                       # Accidental Q-Q edge
        if not nx.is_directed_acyclic_graph(G):
            back = []
            for u, v in G.edges:
                if G.nodes[u]["label"][-1] != "reasoning" or G.nodes[v]["label"][-1] != "reasoning":continue
                if eval(u)[0] > eval(v)[0]: back.append((u, v))
            G.remove_edges_from(back)                                                   # Accidental r - R edge (only 3 examples)
            back = [(u, v) for u, v in G.edges if G.nodes[u]["label"][-1] == "reasoning" and G.nodes[v]["label"][-1] == "question"]
            G.remove_edges_from(back)                                                   # Accidental R-Q edge
        assert nx.is_directed_acyclic_graph(G)
        #for an, su in product(G.ancestors(node), nx.descendants(node)): if G.has_edge(an, su):G.remove_edge(an, su)
        return G

    @staticmethod
    def _strip_context(output, context):
        try: return output[output.rindex(context):]
        except ValueError: return output

    @staticmethod
    def _is_scope_global(formula, cut_point):
        return (
            formula[:cut_point[0]].count("{") == formula[:cut_point[0]].count("}")
            and formula[cut_point[1]:].count("{") == formula[cut_point[1]:].count("}")
        )

    @staticmethod
    def print_graph(graph, mode = "diagram"):
        if mode == "diagram":
            G = graph.reverse()
            q_nodes = [n for n, m in nx.get_node_attributes(graph, 'label').items() if m[1] == "question"]
            layers = {i+1: [j for j in m if j not in q_nodes] for i, m in enumerate(nx.bfs_layers(G, "answer"))}
            layers[max(layers.keys())+1] = q_nodes
            pos = nx.multipartite_layout(graph, align = "horizontal", subset_key = layers)
            nx.draw(graph, pos)
            nx.draw_networkx_labels(graph, pos, labels = nx.get_node_attributes(graph, 'label'))
            plt.show()
        else:
            for line in nx.generate_network_text(graph, sources = ["answer"], vertical_chains = True):
                print(line)

def main():
    #Tracer("math500")()
    Tracer("aime")()

if __name__ == "__main__":
    main()
