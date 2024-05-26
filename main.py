# %%
# matplotlib inline
import os, sys

# windows system：
pwd = os.getcwd()
parent = '\\'.join(pwd.split('\\')[:-1])

# linux system：
# parent = '/'.join(pwd.split('/')[:-1])

sys.path.insert(0, parent)
os.chdir(parent)

# %%
print('-' * 30)
print(os.getcwd())
print('-' * 30)

# %%
import torch
import math
import random
from collections import defaultdict
from DataProcess.DataLoader import CustomedDataset
from DataProcess.AdaptionFunc import adapt_func
from Tools.GlobalGraph import GlobalGraphBuilder
from DataProcess.evaluation import eval_zs_gzsl
# from KnnGraph import KnnGraphBuild
from Tools.LocalGraph import LocalGraphBuilder
from ModelBuilder.GogModel import NetGraph
from DataProcess.graph_batch import negative_generator
from ModelBuilder.VisualModel import AttenFeature
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--nhid', type=int, default=64,
                    help='nhid')
parser.add_argument('--ddi_nhid', type=int, default=2048, # 最终输出的Global Graph的节点特征
                    help='ddi_nhid')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.2,
                    help='dropout ratio')
parser.add_argument('--image_batch_size', type=int, default=50,
                    help='image batch size')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='training ratio')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation ratio')
parser.add_argument('--test_ratio', type=float, default=0.1,
                    help='test ratio')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate for training')
parser.add_argument('--num_epoch', type=int, default=30,
                    help='number of training epoch')
parser.add_argument('--neg_decay', type=float, default=1.0,
                    help='negative sample loss decay')
parser.add_argument('--modular_file', type=str, default='./data/decagon_data/id_SMILE.txt',
                    help='file store the modulars information')
parser.add_argument('--ddi_file', type=str, default='./data/decagon_data/bio-decagon-combo.csv',
                    help='file store the ddi information')
parser.add_argument('--model_path', type=str, default='./saved_model/',
                    help='saved model path')
parser.add_argument('--feature_type', type=str, default='onehot',
                    help='the feature type for the atoms in modulars')
parser.add_argument('--train_type', type=str, default='se',
                    help='training type of the model, each batch contains fixed edges or a side effect graph')
# parser = argparse.ArgumentParser()
args = parser.parse_args()

# %%
idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

# %%
torch.backends.cudnn.benchmark = True

# =======================================================

'''
Load Dataset
加载数据集
'''

dataset = 'CUB'

# Specified dataset location
img_dir = os.path.join(pwd, 'data/{}/'.format(dataset))
file_paths = os.path.join(pwd, 'data/xlsa17/data/{}/res101.mat'.format(dataset))
split_path = os.path.join(pwd, 'data/xlsa17/data/{}/att_splits.mat'.format(dataset))
attr_path = os.path.join(pwd, 'data/attribute/CUB/new_des.csv')
classtxt_path = os.path.join(pwd, 'data/CUB/CUB_200_2011/CUB_200_2011/classes.txt')

# Data Loader
Dataset = CustomedDataset(img_dir, file_paths, split_path, dataset, device)
# PS: shuffle = False! I use the default sequence to construct the graph, so this might be cautious.
train_loader = torch.utils.data.DataLoader(Dataset, batch_size=args.image_batch_size, shuffle=False, num_workers=0)
n_train = len(Dataset) # 图像样本数量
print('Image Samples: ', n_train)


# =======================================================

'''
Specified Settings
参数设置
'''

seed = 214  # 215#
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

dim_f = 2048
dim_v = 300
trainable_w2v = True
image_batch_size = args.image_batch_size
n_epoch = args.num_epoch

normalize_att = Dataset.normalize_att

seen_class = Dataset.seenclasses
unseen_class = Dataset.unseenclasses
class_num = len(seen_class) + len(unseen_class)

img_features = Dataset.data['train_seen']['resnet_features'].to(device) # 原本的2048*7*7 features
img_labels = Dataset.data['train_seen']['labels'].to(device)
att_w2v = Dataset.w2v_att # 312*300
class_embed = Dataset.original_att # 200*312

n_iters = n_train * n_epoch // image_batch_size
report_interval = n_iters // n_epoch

args.num_features = 300 # 决定用词向量
args.device = device
# args.num_edge_features = 128
args.class_num = class_num
threshold = 0.6

# =======================================================

'''
Construct Global Graph...
通过Wordnet构造全局图...调用Tools/GlobalGraph.py
'''

gb = GlobalGraphBuilder(classtxt_path)
gb.build()
globe_G = gb.digitalize()
gb.validation(globe_G) # 输出全局图可视化结果

# =======================================================

'''
Construct Local Graph...
通过Attribute构造属性子图...调用Tools/LocalGraph.py
构造子图输入local_input
'''

local_gb = LocalGraphBuilder(class_embed, att_w2v)
local_input = defaultdict(list) # 构造子图的输入

for idx in range(class_num):
    results = local_gb.build(idx)
    local_input[str(idx)] = list(results)

# =======================================================

'''
Negative Samples Generating...
生成负样本
'''

# 采用多跳寻找最不相关的节点对，提高负样本有效性

neg_pairs = negative_generator(globe_G.edges(), globe_G.number_of_edges())
print('Negative samples generated!')

# =======================================================

'''
Data Process...
将数据适配模型输入形式
'''
# 原本格式：[(0，3)，(1，4)，(2，7)，(3，81)，(4，88)...]
# 要变为： [(0，3)，(3，0), (1，4)，(4，1)，...]

train_edges = adapt_func(globe_G.edges(), device)
n_edges = train_edges.shape[1]
print('Graph Samples: ', n_edges)

args.bach_size = math.ceil(n_edges / (math.ceil(n_train / args.image_batch_size)))


# =======================================================

'''
Model Loading...
加载模型
'''

# GoG Model
model_gog = NetGraph(args).to(args.device)
model_gog.to(device)
optimizer_gog = torch.optim.Adam(model_gog.parameters(), lr=args.learning_rate, weight_decay=5e-4)

# Visual Model
# 单独设置参数
lr = 0.0001
weight_decay = 0.0001  # 0.000#0.#
momentum = 0.9  # 0.#

model_visual = AttenFeature(dim_f, dim_v, att_w2v, class_num, args.ddi_nhid, seen_class, unseen_class, trainable_w2v=False, normalize_V=False)
model_visual.to(device)

params_to_update = []
params_names = []

# Output parameters in Visual Model
for name, param in model_visual.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        params_names.append(name)
        # print("\t", name)

optimizer_visual = torch.optim.RMSprop(params_to_update, lr=lr, weight_decay=weight_decay, momentum=momentum)

# =======================================================

'''
Start Training...
开始训练
'''

for i in range(args.num_epoch):
    batch_num = 0
    flag_num = 0
    last_epoch = False

    print('At {}th epoch'.format(i))

    while not last_epoch:
        model_gog.train()
        model_visual.train()
        optimizer_gog.zero_grad()
        optimizer_visual.zero_grad()

        '''
        Visual Model
        '''

        # visual model inputs and call Visual Model
        batch_label, batch_feature, batch_att = Dataset.next_batch(args.image_batch_size)
        AttenNet_package = model_visual(batch_feature)
        Att_vec = AttenNet_package['AttVec'] # (batch_size, 312, 300) Visual-guided Attribute vectors

        # Update Node feature in Corresponding Local graphs
        # 更新对应属性子图中的节点特征
        # batch_flag用于判断该类节点的属性子图是否已经更新过
        # 问题：对于一个批次内多次出现同一类的情况，应考虑平均（但下列代码仅取第一次进行更新）
        batch_flag = np.zeros(class_num, dtype=bool)
        for i in range(args.image_batch_size):
            if not batch_flag[i]:
                label = batch_label[i].item()
                local_input[str(label)][0] = Att_vec[i]
                batch_flag[i] = True
            else:
                continue

        # Find Corresponding edge samples in the graph
        # 对于一组图片，根据其标签采样对应的 graph edges
        # 问题：仅采样到Global Graph中与叶子节点相连接的边，与graph中高层级部分交互较少，且这些边样本之间没有连接结构
        # 问题：edges样本数量较少，例如图像batch size为50时，由于有重复的类，仅能采样到40+ edge samples
        mask_pos = torch.isin(train_edges, batch_label)
        mask_pos = torch.any(mask_pos, axis=0)
        related_edges = train_edges[:, mask_pos] # 弃用

        '''
        GoG Model
        '''
        # 考虑上述问题，对于每批次图像样本，将整个global graph送入GoG模型，而非选取related_edges送入
        # 每轮批次：对neg_edges进行随机采样，使其数量与global grah的样本数量一致（即每次送入的负样本并不固定）
        neg_sample_pairs = random.sample(neg_pairs, globe_G.number_of_edges())
        neg_train_edges = adapt_func(neg_sample_pairs, device)
        in_data = [local_input, train_edges, neg_train_edges] # gog inputs

        # call GoG model. output: class embeddings
        class_features, loss_gog = model_gog(in_data, globe_G)

        loss_package = AttenNet_package
        loss_package['batch_label'] = batch_label
        loss_package['class_embed'] = class_features

        loss_visual = model_visual.compute_loss(loss_package)

        batch_num += 1
        flag_num += 1

        loss = loss_visual + loss_gog # simply add
        print('Training Loss:', loss)

        loss.backward(retain_graph=True)
        optimizer_gog.step()
        optimizer_visual.step()


    if i <= 0:
        continue



'''
Start Evaluation...
'''

# %%
