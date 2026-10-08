![KisMATH](assets/logo.png)

This repository contains code and data required to reproduce the main results
for the paper titled—"_KisMATH: Do LLMs Have Knowledge of Implicit Structures in
Mathematical Reasoning?_", accepted to the Transactions of the Association for
Computational Linguistics (**TACL**), 14:1308–1328.
[:notebook: Paper](https://aclanthology.org/2026.tacl-1.59/) | [:email: Contact](mailto:soumadeep.saha97@gmail.com)

## Citation

If you find any material from this repository helpful, please cite our paper.

```
@article{saha-etal-2026-kismath,
    title = {{K}is{MATH}: Do {LLM}s Have Knowledge of Implicit Structures in Mathematical Reasoning?},
    author = {Saha, Soumadeep  and Chaturvedi, Akshay  and Saha, Saptarshi  and Garain, Utpal  and Asher, Nicholas},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {14},
    year = {2026},
    address = {Cambridge, MA},
    publisher = {MIT Press},
    url = {https://aclanthology.org/2026.tacl-1.59/},
    doi = {10.1162/tacl.a.729},
    pages = {1308--1328}
}
```

## Abstract

Chain-of-thought (CoT) traces have been shown to improve performance of large
language models on a plethora of reasoning tasks, yet there is no consensus on
the mechanism by which this boost is achieved. To shed more light on this, we
introduce _Causal CoT Graphs_ (**CCGraphs**), which are directed acyclic graphs
automatically extracted from reasoning traces that model finegrained causal
dependencies in language-model outputs.

A collection of **1671** mathematical reasoning problems from MATH500, GSM8K,
and AIME, together with their associated **CCGraphs**, has been compiled into
our dataset—**KisMATH**. Our detailed empirical analysis with 15 open-weight
LLMs shows that (i) reasoning nodes in the **CCGraphs**_ are causal contributors
to the final answer, which we argue is constitutive of reasoning; and (ii) LLMs
emphasize the reasoning paths captured by the _**CCGraphs**_, indicating that
the models internally realize structures similar to our graphs. **KisMATH**
enables controlled, graph-aligned interventions and opens avenues for further
investigation into the role of CoT in LLM reasoning."

## Usage

> [!WARNING]
> The datasets, i.e., extracted **CCGraphs** and **R Paths** can be found
[here](https://huggingface.co/datasets/espressovi/KisMATH).

* The main entry point is ```causality.py```
```
python causaility.py -h
```

* The rank distribution results can be generated using code in
```generate_probability.py```.
```
python generate_probability.py -h
```

## Authors / Contributors

* [Soumadeep Saha](https://espressovi.github.io)
* [Akshay Chaturvedi](https://scholar.google.com/citations?user=28DvXUAAAAAJ&hl=en)
* [Saptarshi Saha](https://openreview.net/profile?id=%7ESaptarshi_Saha1)
* [Utpal Garain](https://isical.ac.in/~utpal)
* [Nicholas Asher](https://www.irit.fr/~Nicholas.Asher/)
