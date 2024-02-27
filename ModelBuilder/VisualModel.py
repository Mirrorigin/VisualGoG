import torch
import torch.nn as nn
import torch.nn.functional as F

class AttenFeature(nn.Module):

    def __init__(self, dim_f, dim_v, w2v_att, class_num, class_fs_dim, seenclass, unseenclass, trainable_w2v=False, normalize_V=False):

        super(AttenFeature, self).__init__()
        self.dim_f = dim_f    # features dimensions
        self.dim_v = dim_v    # word2vectors dimensions
        self.dim_att = w2v_att.shape[0]    # how many attributes (e.g. CUB: 312)
        self.nclass = class_num    # how many classes (e.g. CUB: 200)
        self.hidden = self.dim_att // 2
        self.w2v_att = w2v_att    # Initial Attribute Vector (初始的属性向量)
        self.loss_type = 'CE'
        self.seenclass = seenclass
        self.unseenclass = unseenclass
        self.lambda_ = 0.1

        # Two Layers
        self.linear_1 = nn.Linear(self.dim_f, self.dim_v * 3)
        self.linear_2 = nn.Linear(self.dim_v * 3, self.dim_v)

        if w2v_att is None:
            self.V = nn.Parameter(nn.init.normal_(torch.empty(self.dim_att, self.dim_v)), requires_grad=True)
        else:
            self.w2v_att = F.normalize(w2v_att.clone().detach())
            self.V = nn.Parameter(self.w2v_att.clone(), requires_grad=trainable_w2v)

        # Attention Net
        # trainable parameters
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)  # nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#
        self.W_2 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        # self.W_3 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)

        self.normalize_V = normalize_V
        # Loss函数选择
        self.criterion = nn.CrossEntropyLoss()
        # self.weight_ce = nn.Parameter(torch.eye(self.nclass).float(), requires_grad=False)
        # self.log_softmax_func = nn.LogSoftmax(dim=1)

    def average_pooling(self, input_tensor):
        # 计算平均池化
        output_tensor = torch.mean(input_tensor, axis=-1)
        return output_tensor


    def compute_V(self):
        if self.normalize_V:
            V_n = F.normalize(self.V)
        else:
            V_n = self.V
        return V_n


    def compute_loss(self, in_package):

        # 【待改】：loss设置得比较简单

        image_features = in_package['Img_Fs'] # (batch_size, 2048) 图像全局特征
        class_vectors = in_package['class_embed']
        labels = in_package['batch_label']

        scores = image_features @ class_vectors.t()

        loss = self.criterion(scores, labels)

        print('Visual Model Loss:', loss)

        return loss

    def forward(self, Fs):

        # Fs: (batch_size, 2048, 7, 7)
        shape = Fs.shape
        Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])

        # Initial Attribute Vector (初始的属性向量)
        V_n = self.compute_V()

        '''
        # einstein sum notation
        # b: Batch size \ f: dim feature \ v: dim w2v \ r: number of region \ k: number of classes
        # i: number of attribute \ h : hidden attention dim
        '''

        ## Compute attribute score on each image region
        S = torch.einsum('iv,vf,bfr->bir', V_n, self.W_1, Fs)

        ## compute Dense Attention
        A = torch.einsum('iv,vf,bfr->bir', V_n, self.W_2, Fs)
        A = F.softmax(A, dim=-1)  # compute an attention map for each attribute
        AttFs = torch.einsum('bir,bfr->bif', A, Fs)  # compute attribute-based features
        # AttFs: (batch_size, 312, 2048)

        '''
        Two Linear Layers
        '''

        linear_out = self.linear_1(AttFs)
        Att_vec = self.linear_2(linear_out)
        # Att_vec: (batch_size, 312, 300)

        '''
        Average Pooling
        '''

        Img_fs = self.average_pooling(Fs)

        package = {'AttFs': AttFs, 'AttVec': Att_vec, 'Img_Fs': Img_fs}

        return package