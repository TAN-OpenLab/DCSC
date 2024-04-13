# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2023/6/11 20:51
@Author     : Danke Wu
@File       : TextCNN.py
"""
# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2023/3/28 11:24
@Author     : Danke Wu
@File       : TextCNN.py
"""

import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
from utils import GRL, Mine, MIXUP

import copy
from torch import nn as nn

class sentence_embedding(nn.Module):
    def __init__(self, h_in, h_out):
        super(sentence_embedding, self).__init__()
        self.embedding = nn.Linear(h_in, h_out)
        self.leakrelu = nn.LeakyReLU()
        # self.insnorm = nn.InstanceNorm1d(num_posts,affine=False)

    def forward(self,x, mask_nonzero):

        x = self.embedding(x)
        mask_nonzero_matrix = torch.clone(x)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        x = x - mask_nonzero_matrix.detach()
        x = self.leakrelu(x)
        return x

class Post_Attn(nn.Module):
    def __init__(self,h_in):
        super(Post_Attn, self).__init__()
        self.Attn = nn.Linear(2 * h_in,1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask_nonzero):
        (batch, row) = mask_nonzero
        root = torch.zeros_like(x,device=x.device)
        root[batch,row,:] = x[batch,0,:]

        x_plus = torch.cat([x,root],dim=-1)
        attn = self.Attn(x_plus)
        mask_nonzero_matrix = torch.clone(attn)
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        attn = attn - mask_nonzero_matrix.detach()
        attn.masked_fill_(attn ==0,  float("-inf"))
        attn = self.softmax(attn)
        x = torch.matmul(x.permute(0, 2, 1),attn).squeeze()
        return x, attn


class Reconstruction(nn.Module):
    def __init__(self, filter_num,  emb_dim, f_in, num_posts, window_size, dropout):
        super(Reconstruction, self).__init__()

        self.num_posts = num_posts
        self.fc = nn.Linear(emb_dim, len(window_size) * filter_num, bias= False)
        ### TEXT CNN
        channel_in = 1
        self.window_size = window_size
        self.dconvs = nn.ModuleList([nn.ConvTranspose2d(filter_num, channel_in,  (K, f_in)) for K in self.window_size])

    def forward(self, x,  mask_nonzero):
        x = self.fc(x)
        x_rec = list(x.chunk(len(self.window_size),dim=1))
        size = [self.num_posts - self.window_size[i] + 1 for i in range(len(self.window_size))]
        text = [F.interpolate(x_rec[i].unsqueeze(dim=2), size[i], mode='linear') for i in range(len(self.window_size))]
        text = [F.leaky_relu(self.dconvs[i](text[i].unsqueeze(3))) for i in range(len(self.dconvs))]
        text = torch.cat(text, dim=1)
        text = torch.mean(text, dim =1).squeeze(dim=1)

        mask_nonzero_matrix = torch.clone(text)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        text = text - mask_nonzero_matrix.detach()

        return text


class TextCNN(nn.Module):
    def __init__(self, filter_num, f_in, emb_dim, num_posts, dropout):
        super(TextCNN, self).__init__()

        self.hidden_size = emb_dim

        ### TEXT CNN
        channel_in = 1
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, f_in)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        ## Class  Classifier
        self.class_classifier = nn.Linear( self.hidden_size, 2)


    def forward(self, text):

        ##########CNN##################
        text = text.unsqueeze(dim=1)# add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.fc1(text))

        ### Fake or real
        class_output = self.class_classifier(text)

        return class_output

class TextCNN_stymix(nn.Module):
    def __init__(self, filter_num, f_in, emb_dim, num_posts, dropout):
        super(TextCNN_stymix, self).__init__()

        self.hidden_size = emb_dim

        ### TEXT CNN
        channel_in = 1
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, f_in)) for K in window_size])
        self.fc1 = nn.Sequential(nn.Linear(len(window_size) * filter_num, emb_dim),
                                 nn.LayerNorm(emb_dim),
                                 nn.LeakyReLU())

        self.decoder = Reconstruction(filter_num, emb_dim, f_in, num_posts, window_size, dropout)

        self.disentanglement = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                             nn.Sigmoid())

        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(emb_dim, 2)
        self.cont_rumor_classifier = nn.Linear(emb_dim, 2)
        self.style_classifier = nn.Sequential(nn.Linear(emb_dim, 8, bias=False),
                                              nn.Sigmoid())


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(x.sum(dim=-1), as_tuple=True)
        text = x.unsqueeze(dim=1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text_copy = text.copy()
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        h_all = self.fc1(text)

        dis_weight = self.disentanglement(h_all)
        sty = torch.mul(h_all, dis_weight)
        cont = torch.mul(h_all, 1 - dis_weight)

        stypred_sty = self.style_classifier(sty)
        reverse_cont = self.grad_reversal(cont)
        stypred_cont = self.style_classifier(reverse_cont)

        pred_sty = self.style_rumor_classifier(sty)
        pred_cont = self.cont_rumor_classifier(cont)

        x_rec = self.decoder(cont, mask_nonzero)
        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec




