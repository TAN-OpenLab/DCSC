# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/6/24 20:41
@Author     : Danke Wu
@File       : DCSC_layers.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from Transformer_Encoder import *
from torch.autograd import Variable, Function
from utils import *
import os

class sentence_embedding(nn.Module):
    def __init__(self, h_in, h_out,dropout):
        super(sentence_embedding, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(h_in, h_out),
                                       nn.LayerNorm(h_out),
                                       nn.Dropout(dropout),
                                       nn.LeakyReLU())

    def forward(self,x, mask_nonzero):

        x = self.embedding(x)
        mask_nonzero_matrix = torch.clone(x)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        x = x - mask_nonzero_matrix.detach()
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
        return x, attn

class Content_reconstruction(nn.Module):
    def __init__(self,num_layers, n_head,  h_hid,  e_hid, c_in, dropout):
        super(Content_reconstruction, self).__init__()
        self.decoder = TransformerDecoder(num_layers, n_head, h_hid, e_hid,  dropout)
        self.fc = nn.Linear(h_hid, c_in)
        # self.layernorm = nn.LayerNorm(c_in)

    def forward(self, x, mask_nonzero):

        recov_xc = self.decoder(x, mask_nonzero)
        recov_xc = self.fc(recov_xc)
        # recov_xc = self.layernorm(recov_xc)
        mask_nonzero_matrix = torch.clone(recov_xc)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0,0,:])
        recov_xc = recov_xc - mask_nonzero_matrix.detach()


        return recov_xc

class Disentanglement(nn.Module):
    def __init__(self, c_in):
        super(Disentanglement, self).__init__()
        self.linear = nn.Sequential(nn.Linear(c_in,c_in),
                                    nn.ELU())

    def forward(self, x, mask_nonzero):
        x = self.linear(x)
        mask_nonzero_matrix = torch.clone(x)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        x = x- mask_nonzero_matrix.detach()
        return x


# class DCSC(nn.Module):
#     def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, dropout):
#         super(DCSC, self).__init__()
#         self.embedding = sentence_embedding(c_in, c_hid)
#         self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout)
#
#         self.post_attn = Post_Attn(c_hid, root_enhance=True)
#         self.disentanglement = nn.Sequential(nn.Linear(c_hid, c_hid),
#                                             nn.Sigmoid())
#
#         self.decoder = Content_reconstruction(num_layers_trans,n_head_trans, c_hid, h_hid_trans,
#                                              c_in, dropout)
#
#
#         self.grad_reversal = GRL(lambda_=1)
#
#         self.style_rumor_classifier = nn.Linear(c_hid, 2)
#         self.cont_rumor_classifier = nn.Linear(c_hid, 2)
#         self.style_classifier = nn.Sequential(nn.Linear(c_hid, 8, bias=False),
#                                               nn.Sigmoid())
#
#
#     def forward(self, x, original = True):
#
#         mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
#         h = self.embedding(x, mask_nonzero)
#         h = self.encoder(h, mask_nonzero)
#
#         dis_weight = self.disentanglement(h)
#         sty = torch.mul(h, dis_weight)
#         cont = torch.mul(h, 1-dis_weight)
#
#         sty_all, attn = self.post_attn(sty, mask_nonzero)
#         cont_all, attn = self.post_attn(cont, mask_nonzero)
#
#         B, N, F = sty.size()
#         stypred_sty = self.style_classifier(sty.view(B*N,-1))
#         reverse_cont = self.grad_reversal(cont.view(B*N,-1))
#         stypred_cont = self.style_classifier(reverse_cont)
#
#         pred_sty = self.style_rumor_classifier(sty_all)
#         pred_cont = self.cont_rumor_classifier(cont_all)
#
#         if original:
#             h_mix, x_reidx, mask_nonzero_reidx= MIXUP(invariant=sty, variant=cont, x=x)
#             x_rec = self.decoder(cont, mask_nonzero)
#             x_mix = self.decoder(h_mix, mask_nonzero_reidx)
#             return cont, sty, x_mix, stypred_cont, stypred_sty, pred_cont, pred_sty, x_reidx, x_rec
#         else:
#             x_rec = self.decoder(cont, mask_nonzero)
#             return cont, sty, x_rec, stypred_cont, stypred_sty, pred_cont, pred_sty, x, x_rec

class DCSC(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, num_style, dropout):
        super(DCSC, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout)
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid, c_hid),
                                            nn.Sigmoid())

        self.decoder = Content_reconstruction(num_layers_trans,n_head_trans, c_hid, h_hid_trans,
                                             c_in, dropout)


        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)
        self.style_classifier = nn.Sequential(nn.Linear(c_hid, num_style, bias=False),
                                            nn.Sigmoid())


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1-dis_weight)

        sty_all, s_attn = self.sty_attn(sty, mask_nonzero)
        cont_all, c_attn = self.cont_attn(cont, mask_nonzero)

        stypred_sty = self.style_classifier(sty_all)
        reverse_cont = self.grad_reversal(cont_all)
        stypred_cont = self.style_classifier(reverse_cont)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)

        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims[:-1]:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm2d(1))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, embed_dims[-1]))
            layers.append(torch.nn.BatchNorm2d(1))
            layers.append(torch.nn.LeakyReLU())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, mask_nonzero):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        x = x.unsqueeze(dim=1)
        x = self.mlp(x)
        x = x.squeeze(dim=1)

        mask_nonzero_matrix = torch.clone(x)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        x = x - mask_nonzero_matrix.detach()
        return x

class DCSC_MLP(nn.Module):
    def __init__(self,c_in, c_hid,num_layers_trans, n_head_trans,  h_hid_trans,dropout):
        super(DCSC_MLP, self).__init__()

        self.encoder = MLP(c_in, [c_hid,c_hid], dropout)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid, c_hid),
                                             nn.Sigmoid())
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)
        self.decoder = MLP(c_hid, [c_hid,c_in], dropout)

        # self.decoder = MLP(c_hid[-1], (c_hid[-1],c_in), dropout, True)


        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)
        self.style_classifier = nn.Sequential(nn.Linear(c_hid, 8, bias=False),
                                              nn.Sigmoid())


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)

        h = self.encoder(x,mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1 - dis_weight)

        sty_all, attn = self.sty_attn(sty, mask_nonzero)
        cont_all, attn = self.cont_attn(cont, mask_nonzero)

        stypred_sty = self.style_classifier(sty_all)
        reverse_cont = self.grad_reversal(cont_all)
        stypred_cont = self.style_classifier(reverse_cont)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)

        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec


class DCSC_wogrl(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, num_style, dropout):
        super(DCSC_wogrl, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout)
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid, c_hid),
                                            nn.Sigmoid())

        self.decoder = Content_reconstruction(num_layers_trans,n_head_trans, c_hid, h_hid_trans,
                                             c_in, dropout)

        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)
        self.style_classifier = nn.Sequential(nn.Linear(c_hid, num_style, bias=False),
                                            nn.Sigmoid())


    def forward(self, x, original = True):
        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1 - dis_weight)

        sty_all, attn = self.sty_attn(sty, mask_nonzero)
        cont_all, attn = self.cont_attn(cont, mask_nonzero)

        B, N, F = sty.size()
        stypred_sty = self.style_classifier(sty_all)
        # reverse_cont = self.grad_reversal(cont_all)
        # stypred_cont = self.style_classifier(reverse_cont)
        stypred_cont = 0

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)

        return cont, sty,  stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec


class DCSC_wostyledetection(nn.Module):
    def  __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, num_style, dropout):
        super(DCSC_wostyledetection, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout)
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid, c_hid),
                                            nn.Sigmoid())

        self.decoder = Content_reconstruction(num_layers_trans,n_head_trans, c_hid, h_hid_trans,
                                             c_in, dropout)

        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1-dis_weight)

        sty_all, attn = self.sty_attn(sty, mask_nonzero)
        cont_all, attn = self.cont_attn(cont, mask_nonzero)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        stypred_cont = torch.zeros(1)
        stypred_sty = torch.zeros(1)

        x_rec = self.decoder(cont, mask_nonzero)
        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec



class DCSC_wodecoder(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, num_style, dropout):
        super(DCSC_wodecoder, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout)
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid, c_hid),
                                            nn.Sigmoid())

        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)
        self.style_classifier = nn.Sequential(nn.Linear(c_hid, num_style, bias=False),
                                            nn.Sigmoid())

    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1 - dis_weight)

        sty_all, attn = self.sty_attn(sty, mask_nonzero)
        cont_all, attn = self.cont_attn(cont, mask_nonzero)

        stypred_sty = self.style_classifier(sty_all)
        reverse_cont = self.grad_reversal(cont_all)
        stypred_cont = self.style_classifier(reverse_cont)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        return cont, sty,  stypred_cont, stypred_sty, pred_cont, pred_sty, x


class DCSC_claim(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, num_style, dropout):
        super(DCSC_claim, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout, root_enhance = False)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid,1),
                                            nn.Sigmoid())
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)

        self.decoder = Content_reconstruction(num_layers_trans,n_head_trans, c_hid, h_hid_trans,
                                             c_in, dropout)


        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)
        self.style_classifier = nn.Sequential(nn.Linear(c_hid, num_style, bias=False),
                                            nn.Sigmoid())


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1 - dis_weight)

        sty_all, attn = self.sty_attn(sty, mask_nonzero)
        cont_all, attn = self.cont_attn(cont, mask_nonzero)

        stypred_sty = self.style_classifier(sty_all)
        reverse_cont = self.grad_reversal(cont_all)
        stypred_cont = self.style_classifier(reverse_cont)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)

        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec

class DCSC_interpret(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, num_style, dropout):
        super(DCSC_interpret, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout)
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid, 1),
                                            nn.Sigmoid())
        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)


    def forward(self, x):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1-dis_weight)

        sty_all, attn_sty = self.sty_attn(sty, mask_nonzero)
        pred_sty = self.style_rumor_classifier(sty_all)
        attributions_sty = torch.clone(dis_weight.detach()) # torch.mul(dis_weight, attn_sty)
        attributions_sty = attributions_sty.masked_fill(attn_sty == 0, float("-inf"))
        attributions_sty = torch.softmax(attributions_sty.squeeze(dim=0), dim=0)

        cont_all, attn_cont = self.cont_attn(cont, mask_nonzero)
        pred_cont = self.cont_rumor_classifier(cont_all)
        attributions_cont = torch.clone(1 - dis_weight.detach())# torch.mul(1 - dis_weight, attn_cont)
        attributions_cont = attributions_cont.masked_fill( attn_cont == 0,float("-inf"))
        attributions_cont = torch.softmax(attributions_cont.squeeze(dim=0), dim=0)

        return pred_sty, attributions_sty, pred_cont, attributions_cont


class DCSC_interpret_8styles(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, dropout):
        super(DCSC_interpret_8styles, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout)
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid, 1),
                                            nn.Sigmoid())


        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)

        self.style_classifier = nn.Sequential(nn.Linear(c_hid, 8, bias=False),
                                              nn.Sigmoid())


    def forward(self, x):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1-dis_weight)

        sty_all, attn = self.sty_attn(sty, mask_nonzero)
        pred_sty = self.style_rumor_classifier(sty_all)

        # cont_all, attn = self.cont_attn(cont, mask_nonzero)
        # pred_cont = self.cont_rumor_classifier(cont_all)

        stypred_sty = self.style_classifier(sty_all)


        return  pred_sty, stypred_sty


class DCSC_domaindetection(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, num_domain, dropout):
        super(DCSC_domaindetection, self).__init__()
        self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_hid, h_hid_trans, dropout)
        self.sty_attn = Post_Attn(c_hid)
        self.cont_attn = Post_Attn(c_hid)
        self.disentanglement = nn.Sequential(nn.Linear(c_hid, c_hid),
                                            nn.Sigmoid())

        self.decoder = Content_reconstruction(num_layers_trans,n_head_trans, c_hid, h_hid_trans,
                                             c_in, dropout)


        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(c_hid, 2)
        self.cont_rumor_classifier = nn.Linear(c_hid, 2)
        self.domian_detector = nn.Linear(c_hid, num_domain)


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        h = self.embedding(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1-dis_weight)

        sty_all, s_attn = self.sty_attn(sty, mask_nonzero)
        cont_all, c_attn = self.cont_attn(cont, mask_nonzero)

        stypred_cont = 0
        reverse_sty = self.grad_reversal(sty_all)
        stypred_sty =  self.domian_detector(reverse_sty)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)

        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec


class DCSC_woemb(nn.Module):
    def __init__(self,c_in, c_hid, num_layers_trans, n_head_trans, h_hid_trans, num_style, dropout):
        super(DCSC_woemb, self).__init__()
        # self.embedding = sentence_embedding(c_in, c_hid, dropout)
        self.encoder = TransformerEncoder(num_layers_trans, n_head_trans, c_in, c_hid, dropout)
        self.sty_attn = Post_Attn(c_in)
        self.cont_attn = Post_Attn(c_in)
        self.disentanglement = nn.Sequential(nn.Linear(c_in, c_in),
                                            nn.Sigmoid())

        self.decoder = Content_reconstruction(num_layers_trans,n_head_trans, c_in, c_hid,
                                             c_in, dropout)


        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(c_in, 2)
        self.cont_rumor_classifier = nn.Linear(c_in, 2)
        self.style_classifier = nn.Sequential(nn.Linear(c_in, num_style, bias=False),
                                            nn.Sigmoid())


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        # h = self.embedding(x, mask_nonzero)
        h = self.encoder(x, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1-dis_weight)

        sty_all, s_attn = self.sty_attn(sty, mask_nonzero)
        cont_all, c_attn = self.cont_attn(cont, mask_nonzero)

        stypred_sty = self.style_classifier(sty_all)
        reverse_cont = self.grad_reversal(cont_all)
        stypred_cont = self.style_classifier(reverse_cont)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)

        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec