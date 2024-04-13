# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2023/6/19 9:57
@Author     : Danke Wu
@File       : MDDA.py
"""
import torch.nn as nn
import torch
from utils import GRL


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


class MDDA(nn.Module):
    def __init__(self, f_in, f_hid, num_layers,bidirection):
        super(MDDA, self).__init__()

        self.encoder = Encoder(f_in, f_hid, f_hid,num_layers = num_layers,bidirectional =bidirection)
        self.sty_fc = nn.Sequential(nn.Linear(f_hid,f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())
        self.cont_fc = nn.Sequential(nn.Linear(f_hid, f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())

        self.decoder = Decoder(f_hid*2 , f_hid, f_in,num_layers,bidirection)

        self.grad_reversal = GRL(lambda_=1)

        self.rumor_classifier_sty = nn.Linear(f_hid, 2)
        self.rumor_classifier_cont = nn.Linear(f_hid, 2)

    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(torch.sum(x,dim=-1),as_tuple=True)

        h = self.encoder(x, mask_nonzero)

        cont = self.cont_fc(h)
        style= self.sty_fc(h)

        cont_all = torch.mean(cont, dim=1)
        style_all = torch.mean(style, dim=1)

        rec_x = self.decoder(torch.cat((cont,style), dim=-1), mask_nonzero)

        pred_sty = self.rumor_classifier_sty(style_all )

        cont_reverse = self.grad_reversal(cont_all)
        pred_cont = self.rumor_classifier_cont(cont_reverse)

        return pred_sty, pred_cont, rec_x



class MDDA_stymix_stydetection(nn.Module):
    def __init__(self, f_in, f_hid, num_layers,bidirection):
        super(MDDA_stymix_stydetection, self).__init__()

        self.encoder = Encoder(f_in, f_hid, f_hid,num_layers = num_layers,bidirectional =bidirection)
        self.sty_fc = nn.Sequential(nn.Linear(f_hid,f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())
        self.cont_fc = nn.Sequential(nn.Linear(f_hid, f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())

        self.decoder = Decoder(f_hid*2 , f_hid, f_in,num_layers,bidirection)

        self.grad_reversal = GRL(lambda_=1)

        self.rumor_classifier_sty = nn.Linear(f_hid, 2)
        self.rumor_classifier_cont = nn.Linear(f_hid, 2)

        self.style_classifier = nn.Sequential(nn.Linear(f_hid, 8 ),
                                              nn.Sigmoid())

    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(torch.sum(x,dim=-1),as_tuple=True)

        h = self.encoder(x, mask_nonzero)

        cont = self.cont_fc(h)
        style = self.sty_fc(h)

        cont_all = torch.mean(cont, dim=1)
        style_all = torch.mean(style, dim=1)

        x_rec = self.decoder(torch.cat((cont,style), dim=-1), mask_nonzero)

        pred_sty = self.rumor_classifier_sty(style_all)
        cont_reverse = self.grad_reversal(cont_all)
        pred_cont = self.rumor_classifier_cont(cont_reverse)

        stypred_sty = self.style_classifier(style_all)
        stypred_cont = 0

        return cont, style, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec


class MDDA_stymix_stydetection_grl(nn.Module):
    def __init__(self, f_in, f_hid, num_layers,bidirection):
        super(MDDA_stymix_stydetection_grl, self).__init__()

        self.encoder = Encoder(f_in, f_hid, f_hid,num_layers = num_layers,bidirectional =bidirection)
        self.sty_fc = nn.Sequential(nn.Linear(f_hid,f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())
        self.cont_fc = nn.Sequential(nn.Linear(f_hid, f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())

        self.decoder = Decoder(f_hid*2 , f_hid, f_in,num_layers,bidirection)

        self.grad_reversal = GRL(lambda_=1)

        self.rumor_classifier_sty = nn.Linear(f_hid, 2)
        self.rumor_classifier_cont = nn.Linear(f_hid, 2)

        self.style_classifier = nn.Sequential(nn.Linear(f_hid, 8 ),
                                              nn.Sigmoid())


    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(torch.sum(x,dim=-1),as_tuple=True)

        h = self.encoder(x, mask_nonzero)

        cont = self.cont_fc(h)
        style = self.sty_fc(h)

        cont_all = torch.mean(cont, dim=1)
        style_all = torch.mean(style, dim=1)

        x_rec = self.decoder(torch.cat((cont,style), dim=-1), mask_nonzero)

        pred_sty = self.rumor_classifier_sty(style_all)
        pred_cont = self.rumor_classifier_cont(cont_all)

        stypred_sty = self.style_classifier(style_all)
        cont_reverse = self.grad_reversal(cont_all)
        stypred_cont =  self.style_classifier(cont_reverse)

        return cont, style, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec


class MDDA_stymix_decoder(nn.Module):
    def __init__(self, f_in, f_hid, num_layers,bidirection):
        super(MDDA_stymix_decoder, self).__init__()

        self.encoder = Encoder(f_in, f_hid, f_hid, num_layers=num_layers, bidirectional=bidirection)
        self.sty_fc = nn.Sequential(nn.Linear(f_hid, f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())
        self.cont_fc = nn.Sequential(nn.Linear(f_hid, f_hid),
                                     nn.LayerNorm(f_hid),
                                     nn.LeakyReLU())

        self.decoder = Decoder(f_hid, f_hid, f_in, num_layers, bidirection)

        self.grad_reversal = GRL(lambda_=1)

        self.rumor_classifier_sty = nn.Linear(f_hid, 2)
        self.rumor_classifier_cont = nn.Linear(f_hid, 2)

    def forward(self, x, original=True):
        mask_nonzero = torch.nonzero(torch.sum(x, dim=-1), as_tuple=True)

        h = self.encoder(x, mask_nonzero)

        cont = self.cont_fc(h)
        style = self.sty_fc(h)

        cont_all = torch.mean(cont, dim=1)
        style_all = torch.mean(style, dim=1)

        x_rec = self.decoder(cont, mask_nonzero)

        pred_sty = self.rumor_classifier_sty(style_all)

        cont_reverse = self.grad_reversal(cont_all)
        pred_cont = self.rumor_classifier_cont(cont_reverse)

        stypred_cont = 0
        stypred_sty = 0

        return cont, style, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec


class MDDA_domaindetection(nn.Module):
    def __init__(self, f_in, f_hid, num_layers, bidirection,num_domain):
        super(MDDA_domaindetection, self).__init__()

        self.encoder = Encoder(f_in, f_hid, f_hid, num_layers=num_layers, bidirectional=bidirection)
        self.sty_fc = nn.Sequential(nn.Linear(f_hid, f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())
        self.cont_fc = nn.Sequential(nn.Linear(f_hid, f_hid),
                                     nn.LayerNorm(f_hid),
                                     nn.LeakyReLU())

        self.decoder = Decoder(f_hid * 2, f_hid, f_in, num_layers, bidirection)
        self.grad_reversal = GRL(lambda_=1)

        self.rumor_classifier_sty = nn.Linear(f_hid, 2)
        self.rumor_classifier_cont = nn.Linear(f_hid, 2)

        self.domain_classifier = nn.Linear(f_hid, num_domain)


    def forward(self, x, original = True):
        mask_nonzero = torch.nonzero(torch.sum(x, dim=-1), as_tuple=True)

        h = self.encoder(x, mask_nonzero)

        cont = self.cont_fc(h)
        style = self.sty_fc(h)

        cont_all = torch.mean(cont, dim=1)
        style_all = torch.mean(style, dim=1)

        x_rec = self.decoder(torch.cat((cont, style), dim=-1), mask_nonzero)

        pred_sty = self.rumor_classifier_sty(style_all)
        pred_cont = self.rumor_classifier_cont( cont_all)

        stypred_cont = 0
        style_reverse = self.grad_reversal(style_all)
        stypred_sty = self.domain_classifier(style_reverse)

        return cont, style, stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec



class MDDA_vis(nn.Module):
    def __init__(self, f_in, f_hid, num_layers,bidirection):
        super(MDDA_vis, self).__init__()

        self.encoder = Encoder(f_in, f_hid, f_hid,num_layers = num_layers,bidirectional =bidirection)
        self.sty_fc = nn.Sequential(nn.Linear(f_hid,f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())
        self.cont_fc = nn.Sequential(nn.Linear(f_hid, f_hid),
                                    nn.LayerNorm(f_hid),
                                    nn.LeakyReLU())

        self.decoder = Decoder(f_hid*2 , f_hid, f_in,num_layers,bidirection)

        self.grad_reversal = GRL(lambda_=1)

        self.rumor_classifier_sty = nn.Linear(f_hid, 2)
        self.rumor_classifier_cont = nn.Linear(f_hid, 2)

    def forward(self, x, original = True):

        mask_nonzero = torch.nonzero(torch.sum(x,dim=-1),as_tuple=True)

        h = self.encoder(x, mask_nonzero)

        cont = self.cont_fc(h)
        style= self.sty_fc(h)

        cont_all = torch.mean(cont, dim=1)
        style_all = torch.mean(style, dim=1)

        rec_x = self.decoder(torch.cat((cont,style), dim=-1), mask_nonzero)

        pred_sty = self.rumor_classifier_sty(style_all )

        cont_reverse = self.grad_reversal(cont_all)
        pred_cont = self.rumor_classifier_cont(cont_reverse)

        return cont, style, pred_sty, pred_cont, rec_x
