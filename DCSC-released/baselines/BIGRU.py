# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2023/3/28 21:49
@Author     : Danke Wu
@File       : BIGRU.py
"""

import torch
import torch.nn as nn

from utils import GRL,Mine,MIXUP


class sentence_embedding(nn.Module):
    def __init__(self, h_in, h_out):
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

class Encoder(nn.Module):

    def __init__(self,input_dim, latent_dim,outpuy_dim, num_layers,bidirectional=True):

        super().__init__()
        self.gru = nn.ModuleList([nn.GRU(input_dim, latent_dim, num_layers=1, batch_first=True,
                                         bidirectional=True)] + [
                                     nn.GRU(latent_dim, latent_dim, num_layers=1, batch_first=True,
                                            bidirectional=True) for _ in range(num_layers - 1)])
        self.layernorm = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])


        if bidirectional:
            self.birec = 2
            self.D = num_layers *2
        else:
            self.birec = 1
            self.D= num_layers

        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def forward(self, embedded, mask_nonzero):

        for idx_layer, (gru_layer, norm_layer) in enumerate(zip(self.gru, self.layernorm)):

            embedded, _ = gru_layer(embedded)
            embedded = embedded[:, :, :self.latent_dim] + embedded[:, :, self.latent_dim:]  # sum bidirectional outputs
            embedded = norm_layer(embedded)

        mask_nonzero_matrix = torch.clone(embedded)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        embedded = embedded - mask_nonzero_matrix.detach()

        return embedded


class Decoder(nn.Module):

    def __init__(self, input_dim, latent_dim, output_dim, num_layers=2, bidirectional=True):

        super().__init__()
        self.gru = nn.ModuleList([nn.GRU(input_dim, latent_dim, num_layers=1, batch_first=True,
                                         bidirectional=True)] + [
                                     nn.GRU(latent_dim, latent_dim, num_layers=1, batch_first=True,
                                            bidirectional=True) for _ in range(num_layers - 1)])
        self.layernorm = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        # self.gru = nn.GRU(input_dim, latent_dim,num_layers=num_layers, batch_first=True, bidirectional= bidirectional)
        if bidirectional:
            self.birec = 2
            self.D = num_layers *2
        else:
            self.birec = 1
            self.D= num_layers
        self.out = nn.Linear(latent_dim*self.birec, output_dim)
        self.latent_dim = latent_dim
        self.num_layers =num_layers

        self.h_dim = latent_dim


    def forward(self, embedded, mask_nonzero):
        for idx_layer, (gru_layer, norm_layer) in enumerate(zip(self.gru, self.layernorm)):
            embedded, _ = gru_layer(embedded)
            embedded = embedded[:, :, :self.latent_dim] + embedded[:, :, self.latent_dim:]  # sum bidirectional outputs
            embedded = norm_layer(embedded)
        # output, hidden_n = self.gru(x)
        embedded = self.out(embedded)
        mask_nonzero_matrix = torch.clone(embedded)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        embedded  = embedded - mask_nonzero_matrix.detach()

        return embedded

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


class BIGRU(nn.Module):
    def __init__(self, f_in, f_hid, num_layers,bidirection):
        super(BIGRU, self).__init__()

        # self.sent_emb = sentence_embedding(f_in, f_hid)
        self.encoder = Encoder(f_in, f_hid, f_hid,num_layers = num_layers,bidirectional =bidirection)
        # self.post_attn = Post_Attn(f_hid)

        self.rumor_classifier = nn.Linear(f_hid, 2)


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(torch.sum(x,dim=-1),as_tuple=True)

        # h = self.sent_emb(x, mask_nonzero)
        h = self.encoder(x, mask_nonzero)
        # h_all, attn = self.post_attn(h, mask_nonzero)
        h_all = torch.mean(h, dim=1)

        pred = self.rumor_classifier(h_all)

        return pred


class BIGRU_stymix(nn.Module):
    def __init__(self, f_in, f_hid,num_layers,bidirection):
        super(BIGRU_stymix, self).__init__()

        # self.sent_emb = sentence_embedding(f_in, f_hid)
        self.encoder = Encoder(f_in, f_hid, f_hid,num_layers,bidirection)
        self.post_attn = Post_Attn(f_hid)
        # self.cont_attn = Post_Attn(f_hid)

        self.decoder = Decoder(f_hid , f_hid, f_in,num_layers,bidirection)
        self.disentanglement = nn.Sequential(nn.Linear(f_hid, f_hid),
                                             nn.Sigmoid())

        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(f_hid, 2)
        self.cont_rumor_classifier = nn.Linear(f_hid, 2)
        self.style_classifier = nn.Sequential(nn.Linear(f_hid, 8, bias=False),
                                            nn.Sigmoid())


    def forward(self, x,  original = True):

        mask_nonzero = torch.nonzero(torch.sum(x,dim=-1),as_tuple=True)

        # h = self.sent_emb(x, mask_nonzero)
        h = self.encoder(x, mask_nonzero)

        dis_weight = self.disentanglement(h)
        sty = torch.mul(h, dis_weight)
        cont = torch.mul(h, 1 - dis_weight)

        sty_all, attn = self.post_attn(sty, mask_nonzero)
        cont_all, attn = self.post_attn(cont, mask_nonzero)

        stypred_sty = self.style_classifier(sty_all)
        reverse_cont = self.grad_reversal(cont_all)
        stypred_cont = self.style_classifier(reverse_cont)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)
        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec


class BIGRU_stymix_MINE(nn.Module):
    def __init__(self, f_in, f_hid):
        super(BIGRU_stymix_MINE, self).__init__()

        self.sent_emb = sentence_embedding(f_in, f_hid)
        self.encoder = Encoder(f_hid, [f_hid, f_hid], f_hid)
        self.post_attn = Post_Attn(f_hid)

        self.decoder = Decoder(f_hid, [f_hid, f_hid], f_in)
        self.disentanglement_s = nn.Linear(f_hid, f_hid)
        self.disentanglement_c = nn.Linear(f_hid, f_hid)

        self.grad_reversal = GRL(lambda_=1)

        self.style_rumor_classifier = nn.Linear(f_hid, 2)
        self.cont_rumor_classifier = nn.Linear(f_hid, 2)
        self.style_classifier = nn.Sequential(nn.Linear(f_hid, 8, bias=False),
                                            nn.Sigmoid())

    def forward(self, x, original=True):

        mask_nonzero = torch.where(torch.sum(x, dim=-1), as_tuple=True)

        h = self.sent_emb(x, mask_nonzero)
        h = self.encoder(h, mask_nonzero )

        sty = self.disentanglement_s(h)
        cont = self.disentanglement_c(h)

        sty_all, attn = self.post_attn(sty, mask_nonzero)
        cont_all, attn = self.post_attn(cont, mask_nonzero)

        stypred_sty = self.style_classifier(sty_all)
        # reverse_cont = self.grad_reversal(cont.view(B * N, -1))
        stypred_cont = self.style_classifier(cont_all)

        pred_sty = self.style_rumor_classifier(sty_all)
        pred_cont = self.cont_rumor_classifier(cont_all)

        x_rec = self.decoder(cont, mask_nonzero)
        return cont, sty, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec
