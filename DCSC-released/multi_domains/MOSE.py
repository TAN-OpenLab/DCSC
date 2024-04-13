# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/25 21:55
@Author     : Danke Wu
@File       : MOSE.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskAttention(nn.Module):
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

class MLP(nn.Module):

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

class MoSEModel(torch.nn.Module):
    def __init__(self, emb_dim, num_layers, mlp_dims, dropout, num_expert, num_head):
        super(MoSEModel, self).__init__()
        self.num_expert =  num_expert
        self.num_head = num_head

        input_shape = emb_dim * 2
        expert = []
        for i in range(self.num_expert):
            expert.append(torch.nn.Sequential(nn.LSTM(input_size=mlp_dims[0],
                                                      hidden_size=mlp_dims[0],
                                                      num_layers=num_layers,
                                                      batch_first=True,
                                                      bidirectional=False))
                          )
        self.expert = nn.ModuleList(expert)

        mask = []
        for i in range(self.num_expert):
            mask.append(MaskAttention(mlp_dims[0]))
        self.mask = nn.ModuleList(mask)

        head = []
        for i in range(self.num_head):
            head.append(torch.nn.Linear(mlp_dims[0], 2))
        self.head = nn.ModuleList(head)

        gate = []
        for i in range(self.num_head):
            gate.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, mlp_dims[-1]),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(mlp_dims[-1], self.num_expert),
                                            torch.nn.Softmax(dim=1)))
        self.gate = nn.ModuleList(gate)

        self.rnn = nn.LSTM(input_size=emb_dim,
                           hidden_size=mlp_dims[0],
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=False)
        self.norm = torch.nn.LayerNorm(mlp_dims[0])

        self.attention = MaskAttention(emb_dim)

    def forward(self, x,D):

        mask_nonzero = torch.where(x.sum(dim=-1)==0, 0, 1)
        gate_feature, _ = self.attention(x, mask_nonzero)
        gate_value = []
        for i in range(x.size(0)):
            gate_value.append(self.gate[D[i]](gate_feature[i].view(1, -1)))
        gate_value = torch.cat(gate_value)

        feature, _ = self.rnn(x)
        feature = self.norm(feature)

        rep = 0
        for i in range(self.num_expert):
            tmp_fea, _ = self.expert[i](feature)
            tmp_fea, _ = self.mask[i](tmp_fea, mask_nonzero)
            rep += (gate_value[:, i].unsqueeze(1) * tmp_fea)

        output = []
        for i in range(feature.size(0)):
            output.append(self.head[D[i]](rep[i].view(1, -1)))
        output = torch.cat(output)

        return torch.softmax(output.squeeze(1),dim=-1)