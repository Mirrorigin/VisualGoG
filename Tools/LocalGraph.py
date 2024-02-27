import numpy as np
import torch
import pandas as pd
import networkx as nx
from itertools import combinations
from sklearn.preprocessing import OneHotEncoder

class LocalGraphBuilder:

    def __init__(self, class_embed, att_w2v):

        self.class_prototypes = class_embed
        self.att_vecs = att_w2v
        self.att_num = self.class_prototypes.shape[1]
        # 并未使用全连接，而是设置了一个阈值，将相似度较高的属性之间连接起来
        # 问题：可能存在孤立节点，不与任何其他节点相连
        self.threshold = 0.8

        self.cos_sim = torch.cosine_similarity(self.att_vecs.unsqueeze(1), self.att_vecs.unsqueeze(0), dim=-1)

    # build method
    def build(self, idx):

        prototype = self.class_prototypes[idx]
        nodes_list = torch.nonzero(prototype).squeeze()

        nodes_feature = self.att_vecs

        sub_cos_sim = self.cos_sim[nodes_list, :][:, nodes_list]
        edge_index_original = torch.where(sub_cos_sim > self.threshold)
        edge_index_original = torch.stack(edge_index_original)
        # 剔除自连接的边
        mask = edge_index_original[0, :] != edge_index_original[1, :]
        edge_index = edge_index_original[:, mask]

        edge_index_map = nodes_list[edge_index]

        edge_weights =sub_cos_sim[edge_index[0, :], edge_index[1, :]]

        # 规范化边形式以适应网络输入
        norm_edges, norm_weights = normalized_func(edge_index_map, edge_weights)

        batch_init = torch.zeros(self.att_vecs.shape[0])

        return nodes_feature, norm_edges, norm_weights, batch_init

def normalized_func(edges, weights):

    # 构造子图的边矩阵！
    # 把(0, 1, 2, 3)和(5, 6, 7, 8)分别合为(0, 5, 1, 6, 2, 7, 3, 8)和（5， 0，6，1，7，2，8，3）

    edges_1 = torch.stack((edges[0], edges[1]), dim=1).flatten()
    edges_2 = torch.stack((edges[1], edges[0]), dim=1).flatten()
    normalized_edges = torch.stack((edges_1, edges_2), dim=0)

    normalized_weights = torch.repeat_interleave(weights, 2, dim = 0)

    return normalized_edges, normalized_weights
