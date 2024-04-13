# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/24 14:28
@Author     : Danke Wu
@File       : M3FNED.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.cluster import KMeans

def cal_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim = 1))

def norm(x):
    length = cal_length(x).view(-1, 1)
    x = x / length
    return x

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 2))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1, bias= False)

    def forward(self, inputs, mask_nonzero):
        # print("inputs: ", inputs.shape)
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        # print("scores: ", scores.shape)
        if mask_nonzero is not None:
            scores = scores.masked_fill(mask_nonzero == 0, float("-inf"))

        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        # print("scores: ", scores.shape)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # print("outputs: ", outputs.shape)

        return outputs, scores

class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature

class MemoryNetwork(nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num=10):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)

        self.domain_memory = dict()

    def forward(self, feature, category):
        feature = norm(feature)
        category = torch.unsqueeze(category,dim =-1)
        domain_memory = []
        for i in range(self.domain_num):
            domain_memory.append(self.domain_memory[i])

        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau,
                                                    dim=1)
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)

        domain_att = torch.bmm(sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)).squeeze()

        domain_att = torch.nn.functional.softmax(domain_att * self.tau, dim=1).unsqueeze(1)

        return domain_att

    def write(self, all_feature, category):
        domain_fea_dict = {}
        domain_set = set(category.cpu().detach().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = []
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))

        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)
            topic_att = torch.nn.functional.softmax(
                torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T) * self.tau, dim=1).unsqueeze(2)
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            new_mem = tmp_fea * topic_att
            new_mem = new_mem.mean(dim=0)
            topic_att = torch.mean(topic_att, 0).view(-1, 1)
            self.domain_memory[i] = self.domain_memory[i] - 0.05 * topic_att * self.domain_memory[i] + 0.05 * new_mem

class M3FENDModel(nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, semantic_num, LNN_dim, domain_num,device):
        super(M3FENDModel, self).__init__()
        self.domain_num = domain_num
        self.gamma = 10
        self.memory_num = 10
        self.semantic_num_expert = semantic_num
        self.LNN_dim = LNN_dim
        self.emb_dim = emb_dim

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        content_expert = []
        for i in range(self.semantic_num_expert):
            content_expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.content_expert = nn.ModuleList(content_expert)

        self.gate = nn.Sequential(nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], self.LNN_dim),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        self.weight = torch.nn.Parameter(torch.Tensor(self.LNN_dim,self.semantic_num_expert)).unsqueeze(0).to(device)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.domain_memory = MemoryNetwork(input_dim=self.emb_dim, emb_dim=self.emb_dim,
                                           domain_num=self.domain_num, memory_num=self.memory_num)


        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.all_feature = {}

        self.classifier = MLP(64*5, mlp_dims, dropout)

    def forward(self, x, D ):
        content_masks =  torch.where(x.sum(dim=-1)==0, 0, 1)

        gate_input_feature, _ = self.attention(x, content_masks)
        memory_att = self.domain_memory(gate_input_feature,D)
        domain_emb_all = self.domain_embedder(torch.LongTensor(range(self.domain_num)).cuda())
        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)

        idxs = torch.tensor([index for index in D]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        gate_input = torch.cat([domain_embedding, general_domain_embedding], dim=-1)

        gate_value = self.gate(gate_input).view(x.size(0), 1, self.LNN_dim)

        shared_feature = []
        for i in range(self.semantic_num_expert):
            shared_feature.append(self.content_expert[i](x).unsqueeze(1))

        shared_feature = torch.cat(shared_feature, dim=1)

        embed_x_abs = torch.abs(shared_feature)
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        embed_x_log = torch.log1p(embed_x_afn)

        lnn_out = torch.matmul(self.weight, embed_x_log)
        lnn_exp = torch.expm1(lnn_out)
        shared_feature = lnn_exp.contiguous().view(-1, self.LNN_dim, 320)

        shared_feature = torch.bmm(gate_value, shared_feature).squeeze()

        deep_logits = self.classifier(shared_feature)

        return deep_logits.squeeze(1)

    def save_feature(self, x, D):

        content_masks = torch.where(x.sum(dim=-1)==0, 0, 1)
        content_feature, _ = self.attention(x, content_masks)

        all_feature = norm(content_feature)

        for index in range(all_feature.size(0)):
            domain = int(D[index].cpu().numpy())
            if not (domain in self.all_feature):
                self.all_feature[domain] = []
            self.all_feature[domain].append(all_feature[index].view(1, -1).cpu().detach().numpy())

    def init_memory(self):
        for domain in self.all_feature:
            all_feature = np.concatenate(self.all_feature[domain])
            kmeans = KMeans(n_clusters=self.memory_num, init='k-means++').fit(all_feature)
            centers = kmeans.cluster_centers_
            centers = torch.from_numpy(centers).cuda()
            self.domain_memory.domain_memory[domain] = centers

    def write(self, x, D):

        content_masks =  torch.where(x.sum(dim=-1)==0, 0, 1)
        content_feature, _ = self.attention(x, content_masks)
        all_feature = norm(x)
        self.domain_memory.write(all_feature, D)