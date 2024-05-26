import torch
import random
import numpy as np
from collections import defaultdict, deque


# Using BFS to compute hops between nodes
def bfs(tree, start_node):
    queue = deque([(start_node, 0)])
    visited = {start_node: 0}
    while queue:
        current, hops = queue.popleft()
        for neighbor in tree[current]:
            if neighbor not in visited:
                visited[neighbor] = hops + 1
                queue.append((neighbor, hops + 1))
    return visited

def negative_generator(Global_edges, sample_size):

    '''
    Construct the graph as a TREE
    '''

    # Tips: globe_G中的边均形为[父节点，子节点]
    tree = defaultdict(list)

    for parent, child in Global_edges:
        tree[parent].append(child)
        tree[child].append(parent)

    # Find all leaves (nodes)
    leaves = [node for node, neighbors in tree.items() if len(neighbors) == 1]

    # Saving leaves pairs
    negative_pairs = []
    # 设置为15以上时，无法找到满足对，设置为12时仅能找到18对，设置为11时仅能找到221对
    max_hop = 10 # 可找到1104对

    for i in range(len(leaves)):
        distances = bfs(tree, leaves[i])
        for j in range(i + 1, len(leaves)):
            if distances[leaves[j]] > max_hop:
                negative_pairs.append((leaves[i], leaves[j]))

    if len(negative_pairs) >= sample_size:
        sampled_pairs = random.sample(negative_pairs, sample_size)
    else:
        raise ValueError(f"节点对的数量少于指定数量 {sample_size}。当前数量: {len(negative_pairs)}")

    return sampled_pairs
