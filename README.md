# Readme

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Model code implementation in Zero-shot Learning. Update ing.

## Table of Contents

- [Run](#run)
- [Descriptions](#descriptions)
	- [DataProcess](#dataprocess)
  - [Tools](#tools)
  - [ModelBuilder](#modelbuilder)
- [Related Work](#related-work)


## Run

Simply run the file main.py:

```sh
$ python main.py
```

## Descriptions

A brief introduction to each part of the code.

### DataProcess

Load image dataset and get graph batch.

- graph_batch.py

  Using BFS to find most irrelated nodes and generate negative pairs.

- AdaptionFunc.py

  Fit the data to GoG model input form.

### Tools

- GlobalGraph.py

  Generate Hierarchical Class Graph from Wordnet. (e.g., 285 edges and 286 nodes, where 200 are classes in CUB dataset.)

- LocalGraph.py

  Generate Local Attribute Graph for every class node.

### ModelBuilder

- VisualModel.py

  Including Attention Net and Two Linear Layers.

- GogModel.py

## Related Work

- [GoG] - GoG model
- [DAZLE] - DAZLE model

# VisualGoG

![image](https://github.com/Mirrorigin/VisualGoG/blob/master/Model_View.png)
