from collections import defaultdict
import numpy as np
import torch

'''
将数据适配模型输入形式：
原本input_edges格式：[(0，3)，(1，4)，(2，7)，(3，81)，(4，88)...]
需要变为： [(0，3)，(3，0), (1，4)，(4，1)，...]
'''

def adapt_func(input_edges, device):
    edges_by_nodes = defaultdict(list)

    for u, v in input_edges:
        key = tuple(sorted((u, v)))
        edges_by_nodes[key].append((u, v))
        edges_by_nodes[key].append((v, u))

    sorted_edges = []
    for key, value in sorted(edges_by_nodes.items()):  # 遍历字典中的键值对，按照键的顺序添加到列表中
        sorted_edges.extend(value)  # 这时，列表的内容是：[(0, 999), (999, 0), (1, 206), (206, 1), (1, 899), (899, 1)]

    out_edges = np.array(sorted_edges).T
    out_edges = torch.tensor(out_edges)
    out_edges = out_edges.to(device)

    return out_edges