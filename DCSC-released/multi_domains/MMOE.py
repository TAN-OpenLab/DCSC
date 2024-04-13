# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/25 21:36
@Author     : Danke Wu
@File       : MMOE.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class MMoEModel(torch.nn.Module):
    def __init__(self, emb_dim,  mlp_dims, dropout,expert_num, num_head):
        super(MMoEModel, self).__init__()
        self.num_expert = expert_num
        self.num_head = num_head

        expert = []
        for i in range(self.num_expert):
            expert.append(MLP(emb_dim, mlp_dims, dropout, False))
        self.expert = nn.ModuleList(expert)

        head = []
        for i in range(self.num_head):
            head.append(torch.nn.Linear(mlp_dims[-1], 2))
        self.head = nn.ModuleList(head)

        gate = []
        for i in range(self.num_head):
            gate.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, mlp_dims[-1]),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(mlp_dims[-1], self.num_expert),
                                            torch.nn.Softmax(dim=1)))
        self.gate = nn.ModuleList(gate)

        self.attention = MaskAttention(emb_dim)

    def forward(self, x,D):

        mask_nonzero = torch.where(x.sum(dim=-1)==0, 0, 1)
        feature, _ = self.attention(x, mask_nonzero)

        #每个域单独的门控，训练
        gate_value = []
        for i in range(feature.size(0)):
            gate_value.append(self.gate[D[i]](feature[i].view(1, -1)))
        gate_value = torch.cat(gate_value)

        #根据门控值 将专家结果合并
        rep = 0
        for i in range(self.num_expert):
            rep += (gate_value[:, i].unsqueeze(1) * self.expert[i](feature))

        #每个域对应的输出
        output = []
        for i in range(feature.size(0)):
            output.append(self.head[D[i]](rep[i].view(1, -1)))
        output = torch.cat(output)
        return torch.softmax(output.squeeze(1),dim=-1)