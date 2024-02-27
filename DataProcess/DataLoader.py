# -*- coding: utf-8 -*-
"""
Reference: DAZLE
Load Dataset | CUB & AWA2
"""

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
import numpy as np
import scipy.io as sio
import os, sys
import pickle
import time
from sklearn import preprocessing

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class CustomedDataset(Dataset):

    def __init__(self, img_dir, file_paths, split_paths, dataset, device):
        self.matcontent = sio.loadmat(file_paths)
        self.splitidx = sio.loadmat(split_paths)['trainval_loc']
        self.dataset = dataset
        self.is_balance = False
        if dataset == 'AWA2':
            self.splitidx = self.splitidx - 1 # AWA

        self.image_files = np.squeeze(self.matcontent['image_files'][self.splitidx])
        self.img_dir = img_dir
        self.device = device

        input_size = 224
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.labels = self.matcontent['labels'][self.splitidx].astype(int).squeeze() - 1
        self.read_matdataset()


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        batch_feature = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_label = self.data['train_seen']['labels'][idx].to(self.device)
        # batch_att = self.att[batch_label].to(self.device)

        return (batch_feature, batch_label)

    def next_batch(self, batch_size):
        if self.is_balance:
            idx = []
            n_samples_class = max(batch_size // self.ntrain_class, 1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class), min(self.ntrain_class, batch_size),
                                             replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs, n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]

        batch_feature = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_label = self.data['train_seen']['labels'][idx].to(self.device)
        batch_att = self.att[batch_label].to(self.device)
        return batch_label, batch_feature, batch_att

    def read_matdataset(self):
        path = self.img_dir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print(path)
        print('-' * 30)
        # tic = time.clock()
        tic = time.perf_counter()  # python3.8之后不再支持time.clock()
        hf = h5py.File(path, 'r')
        features = np.array(hf.get('feature_map'))

        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))

        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))

        print('Expert Attr')
        att = np.array(hf.get('att'))
        self.att = torch.from_numpy(att).float().to(self.device)

        original_att = np.array(hf.get('original_att'))
        self.original_att = torch.from_numpy(original_att).float().to(self.device)

        w2v_att = np.array(hf.get('w2v_att'))
        self.w2v_att = torch.from_numpy(w2v_att).float().to(self.device)

        self.normalize_att = self.original_att / 100

        print('Finish loading data in ', time.perf_counter() - tic)

        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]
        test_unseen_feature = features[test_unseen_loc]

        train_feature = torch.from_numpy(train_feature).float()  # .to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature)  # .float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature)  # .float().to(self.device)

        train_label = torch.from_numpy(labels[trainval_loc]).long()  # .to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])  # .long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc])  # .long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        #        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels'] = train_label

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['labels'] = test_unseen_label