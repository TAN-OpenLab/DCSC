# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2023/5/8 16:26
@Author     : Danke Wu
@File       : EDDFN.py
"""
# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import math

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.LayerNorm(embed_dim))
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

class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1, bias= False)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class Grl_func(Function):
    def __init__(self):
        super(Grl_func, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return -lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return Grl_func.apply(x, self.lambda_)


class EDDFNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout):
        super(EDDFNModel, self).__init__()

        self.shared_mlp = MLP(emb_dim, mlp_dims, dropout, False)
        self.specific_mlp = torch.nn.ModuleList([MLP(emb_dim, mlp_dims, dropout, False) for i in range(domain_num)])
        self.decoder = MLP(mlp_dims[-1] * 2, (mlp_dims[-1], emb_dim), dropout, False)
        self.classifier = torch.nn.Linear(2 * mlp_dims[-1], 2)
        self.domain_classifier = nn.Sequential(MLP(mlp_dims[-1], mlp_dims, dropout, False), torch.nn.LeakyReLU(),
                                               torch.nn.Linear(mlp_dims[-1], domain_num))
        self.attention = MaskAttention(emb_dim)
        self.grl = GRL()

    def forward(self, input, D, train =True):

        masks = torch.where(input.sum(dim=-1) == 0, 0, 1)
        bert_feature, _ = self.attention(input, masks)
        specific_feature = []
        for i in range(bert_feature.size(0)):
            specific_feature.append(self.specific_mlp[D[i]](bert_feature[i].view(1, -1)))
        specific_feature = torch.cat(specific_feature)
        shared_feature = self.shared_mlp(bert_feature)
        feature = torch.cat([shared_feature, specific_feature], 1)
        rec_feature = self.decoder(feature)
        output = self.classifier(feature)

        domain_pred = self.domain_classifier(self.grl(shared_feature))
        if not train:
            for i in range(bert_feature.size(0)):
                specific_feature_n = []
                for specific_mlp in self.specific_mlp:
                    specific_feature_n.append(specific_mlp(bert_feature[i].view(1, -1)))
                specific_feature_n  = torch.cat(specific_feature_n)
                specific_feature_n = specific_feature_n.mean(0)
                specific_feature.append(specific_feature_n)
            specific_feature = torch.cat(specific_feature)
            feature = torch.cat([shared_feature,specific_feature], 1)
            output = self.classifier(feature)

        return output.squeeze(1), rec_feature, bert_feature, domain_pred