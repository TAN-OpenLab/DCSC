# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2023/3/31 15:04
@Author     : Danke Wu
@File       : TransE.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from Transformer_Encoder import *
from torch.autograd import Variable, Function
from utils import *

class sentence_embedding(nn.Module):
    def __init__(self, h_in, h_out,dropout):
        super(sentence_embedding, self).__init__()
        self.embedding = nn.Linear(h_in, h_out)
        self.leakrelu = nn.LeakyReLU()

    def forward(self,x, mask_nonzero):

        x = self.embedding(x)
        mask_nonzero_matrix = torch.clone(x)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        x = x - mask_nonzero_matrix.detach()
        x = self.leakrelu(x)
        return x

class Post_Attn(nn.Module):
    def __init__(self,h_in,root_enhance=False):
        super(Post_Attn, self).__init__()
        self.root_enhance = root_enhance
        if root_enhance :

            self.Attn = nn.Linear(2 * h_in, 1)
        else:
            self.Attn = nn.Linear(h_in, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask_nonzero):

        (batch, row) = mask_nonzero
        if self.root_enhance:
            root = torch.zeros_like(x, device=x.device)
            root[batch, row, :] = x[batch, 0, :]

            x_plus = torch.cat([x, root], dim=-1)
            attn = self.Attn(x_plus)
        else:
            attn = self.Attn(x)

        mask_nonzero_matrix = torch.clone(attn)
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        attn = attn - mask_nonzero_matrix.detach()
        attn.masked_fill_(attn ==0, -1e20)
        attn = self.softmax(attn)
        x = torch.matmul(x.permute(0, 2, 1),attn).squeeze()
        return x

class Content_reconstruction(nn.Module):
    def __init__(self,num_layers, n_head,  h_hid,  e_hid, c_in, dropout, pos_embed=None):
        super(Content_reconstruction, self).__init__()
        self.decoder = TransformerDecoder(num_layers, n_head, h_hid, e_hid,  dropout, pos_embed)
        self.fc = nn.Sequential(nn.Linear(h_hid, c_in),
                                nn.LeakyReLU())
        # self.layernorm = nn.LayerNorm(c_in, elementwise_affine= False)

    def forward(self, x, mask_nonzero):

        recov_xc = self.decoder(x, mask_nonzero)
        recov_xc = self.fc(recov_xc)
        mask_nonzero_matrix = torch.clone(recov_xc)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0,0,:])
        recov_xc = recov_xc - mask_nonzero_matrix.detach()
        # recov_xc = self.layernorm(recov_xc)

        return recov_xc


class TransE(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, dropout):
        super(TransE, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid,dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid,h_hid_trans, dropout)
        self.post_attn = Post_Attn(c_hid)

        self.rumor_classifier = nn.Linear(c_hid, 2)


    def forward(self, x):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)
        h = self.post_attn(h, mask_nonzero)

        pred = self.rumor_classifier(h)

        return pred

class TransE_stymix(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans,dropout):
        super(TransE_stymix, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid,h_hid_trans, dropout)
        self.sty_attn = Post_Attn( c_hid)
        self.cont_attn = Post_Attn( c_hid)
        self.disentanglement = nn.Sequential(nn.Linear( c_hid, c_hid),
                                            nn.Sigmoid())

        self.decoder = Content_reconstruction(num_layers_trans,n_head_trans, c_hid,h_hid_trans,
                                             c_hid, dropout)


        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear( c_hid, 2)
        self.cont_rumor_classifier = nn.Linear( c_hid, 2)
        self.style_classifier = nn.Sequential(nn.Linear( c_hid,8, bias=False),
                                            nn.Sigmoid())


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        # h = self.embedding(x, mask_nonzero)
        h = self.encoder(x, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1-dis_weight)

        sty_all= self.sty_attn(sty, mask_nonzero)
        cont_all = self.cont_attn(cont, mask_nonzero)

        stypred_sty = self.style_classifier(sty_all)
        reverse_cont = self.grad_reversal(cont_all)
        stypred_cont = self.style_classifier(reverse_cont)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)

        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec