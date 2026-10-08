#!/bin/python3.13
import os, toml, re, json
from jinja2 import Template
from copy import deepcopy
from openai import OpenAI
from tqdm import tqdm
import networkx as nx
import numpy as np
import sympy
from sympy.parsing.latex import parse_latex
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from TextNumbers import TextNumbers
from collections import deque

config = toml.load("config.toml")["display"]
plt.rcParams.update({
    'text.usetex': True,
    'font.size': config["font_size"],
    'font.family': config["font_family"],
    'text.latex.preamble' : r'\usepackage{amsmath}',
})


class Draw:
    delY = config["delY"]
    max_len = config["max_len"]
    line_spacing = config["line_spacing"]
    min_sp = config["min_word_spacing"]
    def __call__(self, G, **kwargs):
        self.G = G
        self.question = kwargs["question"]
        self.reasoning = kwargs["reasoning"]
        self.answer = kwargs["answer"]
        self.keyargs = kwargs
        self._init_figure()
        self.draw()
        self.draw_boxes()
        self.draw_arrows()
        self.fig.tight_layout()
        self.fig.savefig(self.keyargs["output_file"], format = "pdf", bbox_inches = "tight")
        plt.close('all')

    def draw(self):
        question = self.get_par("question", self.question)
        self.place_paragraph(question)
        reasoning = self.get_par("reasoning", self.reasoning)
        self.place_paragraph(reasoning)
        answer = [
            {"text": "Answer:", "length":self._get_size("Answer:"), "node":None},
            {"text": self.answer, "length":self._get_size(self.answer), "node":"answer"},
        ]
        self.place_paragraph(answer)

    def get_par(self, part, text):
        q_nodes = [eval(n) for n, m in nx.get_node_attributes(self.G, 'label').items() if m[1] == part]
        q_nodes = sorted(q_nodes, key = lambda x: x[0])
        word_list = []
        start = 0
        for l, r in q_nodes:
            for word in re.split(r"\s+", re.sub(r"(?<!\\)\$", "", text[start:l]).strip()):
                word = self._sanitize(word)
                word_list.append({"text": word, "length": self._get_size(word), "node": None})
            word = self._text_to_formula(text[l:r])
            word_list.append({"text": word, "length": self._get_size(word), "node": (l, r, part[0])})
            start = r
        for word in re.split(r"\s+", re.sub(r"(?<!\\)\$", "", text[start:]).strip()):
            word = self._sanitize(word)
            word_list.append({"text": word, "length": self._get_size(word), "node": None})
        return word_list

    def place_paragraph(self, word_list):
        line, lengths, tag, heights = [], [], [], []
        for word in word_list:
            line.append(word["text"])
            lengths.append(word["length"][0])
            heights.append(word["length"][1])
            tag.append(word["node"])
            if np.sum(lengths) + self.min_sp*(len(lengths)-1) > self.max_len:
                min_sp = (self.max_len - np.sum(lengths))/(len(line) - 1)
                for w, l, h, tt in zip(line, lengths, heights, tag):
                    if tt is not None: self.pos[tt] = (float(self.X), self.Y, l, h)
                    self.ax.text(self.X, self.Y, w)
                    self.X = self.X + l/self.max_len + min_sp/self.max_len
                heights, lengths, line, tag = [], [], [], []
                self.Y = self.Y + self.line_spacing*self.delY
                self.X = 0
        if line:
            for w, l, h, tt in zip(line, lengths, heights, tag):
                if tt is not None: self.pos[tt] = (float(self.X), self.Y, l, h)
                self.ax.text(self.X, self.Y, w)
                self.X = self.X + l/self.max_len + self.min_sp/self.max_len
            self.Y = self.Y + self.line_spacing*self.delY
        self.Y = self.Y + self.line_spacing*self.delY
        self.X = 0

    def draw_boxes(self):
        for x, y, w, h in self.pos.values():
            rr = patches.Rectangle((x, y - h/(2*self.max_len)), w/self.max_len, 2*h/self.max_len, linewidth = 1, color = (0.706, 1, 0.98, 0.46))
            self.ax.add_patch(rr)

    def draw_arrows(self):
        all_ys = np.array(list(set([v[1] for v in self.pos.values()])))
        for u, v in self.G.edges:
            (s, t) = (eval(u), eval(v)) if v != "answer" else (eval(u), v)
            sp = (s[0], s[1], self.G.nodes[u]["label"][1][0])
            tp = "answer" if v == "answer" else (t[0], t[1], self.G.nodes[v]["label"][1][0])
            sx, sy, sw, sh = self.pos[sp]
            tx, ty, tw, th = self.pos[tp]
            sw, tw = float(sw/self.max_len), float(tw/self.max_len)
            sh, th = float(2*sh/self.max_len), float(2*th/self.max_len)
            if ty == sy:
                path = matplotlib.path.Path(
                    [(sx + sw/2, sy + sh/1.5), (sx + sw/2, sy + sh/1.5 + 0.03),
                     (tx + tw/2, sy + sh/1.5 + 0.03), (tx + tw/2, sy + th/1.5)],
                    [1, 2, 2, 2], closed = False
                )
                ar = patches.PathPatch(path, facecolor=None, fill = None, lw=0.7, ls = "--", alpha = 0.3)
                self.ax.add_patch(ar)
            else:
                mps = [sy + f*(ty + th/1.5 - sy - sh/3) for f in np.linspace(0, 1, 10)]
                midp = max(mps, key = lambda x:np.min(np.abs(all_ys - x)))
                path = matplotlib.path.Path(
                    [(sx + sw/2, sy - sh/3), (sx + sw/2, midp),
                    (tx + tw/2, midp), (tx + tw/2, ty + th/1.5)],
                    [1, 2, 2, 2], closed = False
                )
                ar = patches.PathPatch(path, color= np.random.rand(3), facecolor=None, fill = None, lw=0.7, ls="--", alpha = 0.5)
                self.ax.add_patch(ar)

    def _get_size(self, text):
        dummy = self.dax.text(0, 0, text)
        bb = dummy.get_window_extent(self.dummy.canvas.get_renderer())
        return bb.width, bb.height

    def _sanitize(self, text):
        text = re.sub(r"\\text\{", "", text)
        text = re.sub(r"\}", "", text)
        text = re.sub(r"(?<!\\)\$(?!.*(?<!\\)\$)([^\n$]*)", r"$\1$", text)
        text = re.sub(r'(\\(?:Long)?[rR]ightarrow|\\[lL]eftarrow|\\implies|\\leq?|\\geq?|\\times)', r'$\1$', text)
        return text

    def _text_to_formula(self, text):
        if re.match(r"^\w+$", text):
            return text
        text = re.sub(r"dfrac", "frac", text)
        text = re.sub(r"tfrac", "frac", text)
        text = re.sub(r"%", "\\%", text)
        return f"${text}$"

    def _init_figure(self):
        self.fig = plt.figure(figsize = config["size"], dpi = config["dpi"])
        self.dummy = plt.figure(figsize = config["size"], dpi = config["dpi"])
        self.ax = self.fig.add_subplot()
        self.dax = self.dummy.add_subplot()
        self.dax.axis('off')
        self.ax.axis('off')
        self.X, self.Y = 0, 0.95
        self.pos = {}
