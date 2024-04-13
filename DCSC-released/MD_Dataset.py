# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/14 10:14
@Author     : Danke Wu
@File       : MD_Dataset.py
"""
import os, pickle
import torch
import torch.utils.data as Data

class MyDataset_withdomains(Data.Dataset):
    def __init__(self, filepath, file_list, num_nodes, domian_dict):
        # 获得训练数据的总行
        self.filepath = filepath
        self.file_list = file_list
        self.number = len(self.file_list)
        self.num_nodes = num_nodes
        self.domain_dict = domian_dict

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        file = self.file_list[idx]
        #X: content matirx (N,768) ,XS:node degree matrix(N,3), A: adjection matrix (N,N) ,T: public _time_invterval(N,1) , y:label 0or1
        X, y, domain_label = pickle.load(open(os.path.join(self.filepath, file), 'rb'), encoding='utf-8')

        self.number = len(X)

        X = torch.tensor(X, dtype=torch.float32)
        if domain_label in ['health_deterrent_cancer', 'health_deterrent_diabetes']:
            domain_label = 'health'
        domain_label = torch.tensor(self.domain_dict[domain_label], dtype= torch.long)

        if torch.sum(torch.isinf(X)) >0 or torch.sum(torch.isnan(X))>0:
            print(file)

        #early detection
        if (X.size()[0] > self.num_nodes):
            X = X[:self.num_nodes, :]
        y = int(y)
        y = torch.tensor(y, dtype=torch.long)
        N, F = X.size()
        if (X==0).sum() == F*N :
            print(os.path.join(self.filepath, file))
        if torch.sum(X.sum(-1))==X.size()[0]:
            print(os.path.join(self.filepath, file))

        return X, y, domain_label