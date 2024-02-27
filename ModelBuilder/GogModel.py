import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv, SAGPooling, NNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class NetGraph(torch.nn.Module):

    def __init__(self, args):
        super(NetGraph, self).__init__()
        self.args = args
        self.num_features = args.num_features
        # self.ddi_num_features = args.ddi_num_features
        # self.num_edge_features = args.num_edge_features
        self.nhid = args.nhid
        self.ddi_nhid = args.ddi_nhid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid).to(args.device)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv2 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv3 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)

        self.conv_noattn = GCNConv(6 * self.nhid, self.ddi_nhid).to(args.device)

        # self.nn = torch.nn.Linear(self.num_edge_features, 6 * self.nhid * self.ddi_nhid)
        # self.conv4 = NNConv(6 * self.nhid, self.ddi_nhid, self.nn).to(args.device)

        self.lin1 = torch.nn.Linear(self.ddi_nhid, self.ddi_nhid)
        self.lin2 = torch.nn.Linear(self.ddi_nhid, self.ddi_nhid)

    # 该函数用来基于叶子节点特征获取其它节点的平均特征
    def Get_Fs(self, Digraph, children_fs):

        Fs_dict = {}

        for i, row in enumerate(children_fs):
            Fs_dict[i] = row

        # 使用拓扑排序，保证在更新一个节点的特征之前，其所有子节点的特征都已经被更新
        for node in reversed(list(nx.topological_sort(Digraph))):
            if Digraph.out_degree(node) > 0:  # 如果节点不是叶子节点
                children_features = [Fs_dict[child] for child in Digraph.successors(node)]
                Fs_dict[node] = torch.stack(children_features).mean(dim=0)

        Graph_Fs = torch.stack([Fs_dict[key] for key in sorted(Fs_dict.keys())])

        return Graph_Fs


    def forward(self, data, Digraph):

        # modular_data：200个类节点对应的子图数据
        modular_data, global_edge_index, neg_edge_index = data
        modular_output = []

        '''
        对所有子图进行卷积
        '''

        ids = list(modular_data.keys())
        for modular_id in ids:
            x, edge_index, edge_weight, batch = modular_data[modular_id]
            x = x.to(self.args.device)
            edge_index = edge_index.to(self.args.device)
            edge_weight = edge_weight.to(self.args.device)
            batch = batch.to(self.args.device)

            x = F.relu(self.conv1(x, edge_index, edge_weight))
            batch = batch.long()
            x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, edge_weight, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x, edge_index, edge_weight, batch, _, _ = self.pool2(x, edge_index, edge_weight, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index, edge_weight))
            x, edge_index, edge_weight, batch, _, _ = self.pool3(x, edge_index, edge_weight, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            out_x = torch.cat((x1, x2, x3), dim=1)
            modular_output.append(out_x)

        '''
        全局图卷积
        '''
        # modular_output是长度为class num（200）的tensor list，每个tensor形为 (1, nhid*6 = 384)
        modular_feature = torch.cat(tuple(modular_output))
        modular_feature = nn.Dropout(self.args.dropout_ratio)(modular_feature)
        class_num = modular_feature.shape[0]

        # 上述modular_feature仅为全局图中叶子节点的node feature，形为(200, 384)
        # 通过自定义函数得到全局图中所有结点的node feature，形为(286, 384)
        graph_feature = self.Get_Fs(Digraph, modular_feature)


        x = F.relu(self.conv_noattn(graph_feature, global_edge_index))
        # 切片tensor，作为class embedding
        class_embedding = x[:class_num]

        pos_source, pos_target, neg_source, neg_target = self.feature_split(x, global_edge_index, neg_edge_index)
        # sigmoid or softmax or nothing, add relu
        pos_feat_x = self.lin1(pos_source)
        pos_feat_y = self.lin2(pos_target)
        neg_feat_x = self.lin1(neg_source)
        neg_feat_y = self.lin2(neg_target)

        norm_pos, norm_neg = self.xent_loss(pos_feat_x, pos_feat_y, neg_feat_x, neg_feat_y)
        pos_tgt = torch.ones_like(norm_pos)
        neg_tgt = torch.zeros_like(norm_neg)

        loss = nn.BCEWithLogitsLoss()(norm_pos, pos_tgt) + nn.BCEWithLogitsLoss()(norm_neg, neg_tgt)
        print('Gog Model Loss:', loss)

        # return class_embedding, loss, norm_pos, norm_neg, pos_feat_x, x
        return class_embedding, loss

    def feature_split(self, features, edge_index, neg_index):
        # while doing the prediction, maybe we can only use half features.
        source, target = edge_index
        pos_source = features[source]
        pos_target = features[target]
        source, target = neg_index
        neg_source = features[source]
        neg_target = features[target]

        return pos_source, pos_target, neg_source, neg_target

    def xent_loss(self, pos_x, pos_y, neg_x, neg_y):
        pos_score = torch.sum(torch.mul(pos_x, pos_y), 1)
        neg_score = torch.sum(torch.mul(neg_x, neg_y), 1)

        return pos_score, neg_score