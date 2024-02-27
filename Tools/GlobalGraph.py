# Generate Global Graph from wordnet

from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib.pyplot as plt
import nltk

nltk.download('wordnet')

class GlobalGraphBuilder:
    def __init__(self, classtxt_path):
        self.classes = []
        self.G = nx.DiGraph()
        self.target = 'bird'  # CUB的类全部都是bird
        self.node_to_number = {}

        with open(classtxt_path) as f:
            for line in f:
                parts = line.strip().split('.')
                c_name = parts[-1]
                c_name = c_name.replace(' ', '_')
                self.classes.append(c_name.casefold())

    def get_hypernyms(self, synset, target_synset, path=None):
        if path is None:
            path = [synset.name().split('.')[0]]
        else:
            path = [synset.name().split('.')[0]] + path

        if synset == target_synset: # 找到目标上位词才返回
            return path

        else:
            hypernyms = synset.hypernyms()  # 获取上位词
            if len(hypernyms) == 0:
                return []
            else:
                return [p for h in hypernyms for p in self.get_hypernyms(h, target_synset, path)]  # 递归

    def get_all_hypernyms(self, word, target_word):
        target = wn.synset(target_word + '.n.01')
        synsets = wn.synsets(word)
        for s in synsets:
            path = self.get_hypernyms(s, target)
        if len(path) and path[-1] != word: # 某些情况下可能直接用同义词取代原词语，例如gray_catbird会被catbird取代
            path[-1] = word
        return path

    def build(self):

        all_paths = []
        invalid_word = []
        valid_word = []

        for c in self.classes:
            if len(wn.synsets(c)) > 0:  # 判断单词在不在wordnet里面
                path = self.get_all_hypernyms(c, self.target)
                if len(path):
                    all_paths.append(path)
                    # print('->'.join([p for p in path]))
                    continue
                else:
                    invalid_word.append(c)
            else:
                invalid_word.append(c)

        for p in all_paths:
            for i in range(len(p) - 1):
                self.G.add_edge(p[i], p[i + 1])

        # 第二步：处理invalid word

        # 将单词拆分，判断拆分的单词是否在已有节点中
        for word in invalid_word:
            split_word = word.split('_')
            for w in split_word:
                if w.casefold() in self.G.nodes:
                    self.G.add_edge(w, word)
                    valid_word.append(word)
                    break

        invalid_word = [x for x in invalid_word if x not in valid_word]

        new_paths = []
        valid_word = []

        # 第三步：对余下节点继续判断拆分的单词是否在wordnet中
        for word in invalid_word:
            split_word = word.split('_')
            for w in split_word:
                if len(wn.synsets(w)) > 0:
                    path = self.get_all_hypernyms(w, self.target)
                    if len(path):
                        path = path + [word.casefold()]
                        new_paths.append(path)
                        valid_word.append(word)
                        break

        for p in new_paths:
            for i in range(len(p) - 1):
                self.G.add_edge(p[i], p[i + 1])

        invalid_word = [x for x in invalid_word if x not in valid_word]

        for word in invalid_word:
            self.G.add_edge('bird', word.casefold())

        return self.G

    def digitalize(self):

        counter = 0

        for c in self.classes:
            if c.casefold() in self.G.nodes:
                self.node_to_number[c] = counter
                counter += 1

        for node in self.G.nodes:
            if node not in self.node_to_number:
                self.node_to_number[node] = counter
                counter += 1

        # print(self.node_to_number)
        digit_graph = nx.relabel_nodes(self.G, self.node_to_number)

        return digit_graph

    def validation(self, graph):
        # 验证：计算 outdegree 为 0 的节点数量并打印
        # 此处数量应与类数量对应，即200
        outdegrees = dict(graph.out_degree())
        zero_outdegree_nodes = sum(1 for node, outdegree in outdegrees.items() if outdegree == 0)

        print(f"Number of nodes with outdegree 0: {zero_outdegree_nodes}")

        plt.figure(figsize=(20, 20))
        nx.draw(self.G, with_labels=True, font_size=10, node_size=1000, node_color='lightblue', edge_color='gray')

        plt.show()