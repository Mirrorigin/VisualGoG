import torch
import random
import numpy as np

def batch_feed(batch_size, batch_num, train_edges, neg_edges):
    # note that the batch_num should start from 0
    last_epoch = False
    # 判断当前的批次编号是否超过了训练数据的总数
    if batch_size * (batch_num + 1) > train_edges.size(1):
        start_index = train_edges.size(1) - batch_size
        end_index = train_edges.size(1)
        last_epoch = True
    else:
        start_index = batch_size * batch_num
        end_index = batch_size * (batch_num + 1)

    t_batch = train_edges[:, start_index: end_index]
    n_batch = neg_edges[:, start_index: end_index]

    return t_batch, n_batch, last_epoch

def is_member(edge_pair, edges_list):
    y = np.where(edges_list[0] == edge_pair[0])[0]

    if len(y) > 0: # 如果坐标都不为空，表明元素存在
        for i in y:
            if edges_list[1][i] == edge_pair[1]: # 判断对应的另一维
                return True
    return False

'''

'''
def negative_generator(positive_edges, nodes_num):

    positive_edges = np.array(positive_edges.cpu())

    neg_train_edges_col = []
    neg_train_edges_row = []

    for i in range(len(positive_edges[0])):
        if i % 2 == 1:
            continue
        node_1 = positive_edges[0][i]
        node_2 = random.randint(0, nodes_num-1)
        count = 0
        while node_1 == node_2 or is_member([node_1, node_2], positive_edges) or \
                is_member([node_1, node_2], np.array([neg_train_edges_col, neg_train_edges_row])): # 注意这里一定要转为ndarray的形式，否则在is_member中会报错
            if count >= 1000:
                node_1 = random.randint(0, nodes_num - 1)
                count = 0
            node_2 = random.randint(0, nodes_num - 1)
            count += 1
        neg_train_edges_col.append(node_1)
        neg_train_edges_row.append(node_2)
        neg_train_edges_col.append(node_2)
        neg_train_edges_row.append(node_1)

    return torch.LongTensor([neg_train_edges_col, neg_train_edges_row])