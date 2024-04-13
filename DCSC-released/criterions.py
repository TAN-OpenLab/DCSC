# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/10/23 9:21
@Author     : Danke Wu
@File       : criterions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from  typing import *
import torch_scatter
import random



class Triple_Loss(nn.Module):
    def __init__(self, margin, domain_num):
        super(Triple_Loss,self).__init__()
        # self.loss = nn.L1Loss()
        self.domain_num =  domain_num
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.relu = nn.ReLU()
        self. margin = margin


    def forward(self, input, D):
        posi, nega = torch.chunk(input, 2, dim=0)
        posi_D, nega_D = torch.chunk(D, 2, dim=0)

        #intra-domian
        posi_anchor = torch.softmax(torch_scatter.scatter_mean(posi, posi_D,dim_size=self.domain_num, dim=0),dim=-1)
        nega_anchor = torch.softmax(torch_scatter.scatter_mean(nega, nega_D,dim_size=self.domain_num, dim=0),dim=-1)
        posi = torch.softmax(posi, dim=-1)
        nega = torch.softmax(nega, dim=-1)


        pp_anchor_mean = torch.repeat_interleave(posi_anchor.mean(dim=0, keepdim =True), posi.size(0), dim=0)
        nn_anchor_mean = torch.repeat_interleave(nega_anchor.mean(dim=0, keepdim=True), nega.size(0), dim=0)
        pn_anchor_mean = torch.repeat_interleave(posi_anchor.mean(dim=0, keepdim=True), nega.size(0), dim=0)
        np_anchor_mean = torch.repeat_interleave(nega_anchor.mean(dim=0, keepdim=True), posi.size(0), dim=0)

        loss_d = self.relu(self.margin + self.loss(posi.log(), pp_anchor_mean.detach()) - self.loss(posi.log(), np_anchor_mean.detach()) ) + \
                 self.relu(self.margin + self.loss(nega.log(), nn_anchor_mean.detach()) - self.loss(nega.log(), pn_anchor_mean.detach()) )


        return  loss_d


class Quadruple_Loss(nn.Module):
    def __init__(self, margin, domain_num):
        super(Quadruple_Loss,self).__init__()
        # self.loss = nn.L1Loss()
        self.domain_num =  domain_num
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.relu = nn.ReLU()
        self. margin = margin


    def forward(self, input, D):
        posi, nega = torch.chunk(input, 2, dim=0)
        posi_D, nega_D = torch.chunk(D, 2, dim=0)

        #intra-domian
        posi_anchor = torch.softmax(torch_scatter.scatter_mean(posi, posi_D,dim_size=self.domain_num, dim=0),dim=-1)
        nega_anchor = torch.softmax(torch_scatter.scatter_mean(nega, nega_D,dim_size=self.domain_num, dim=0),dim=-1)

        # inter-domian
        posi_anchor_mean = torch.repeat_interleave(posi_anchor.mean(dim=0, keepdim =True), posi_anchor.size(0), dim=0)
        nega_anchor_mean = torch.repeat_interleave(nega_anchor.mean(dim=0, keepdim=True), nega_anchor.size(0), dim=0)

        loss_d = self.relu(self.margin + self.loss(posi_anchor.log(), posi_anchor_mean.detach()) + self.loss(nega_anchor.log(), nega_anchor_mean.detach())
                           - (self.loss(posi_anchor_mean.log(), nega_anchor_mean) +self.loss(nega_anchor_mean.log(), posi_anchor_mean))/2)

        return loss_d


class Multilevel_Contrastive_Loss(nn.Module):
    def __init__(self, margin, domain_num):
        super(Multilevel_Contrastive_Loss,self).__init__()
        # self.loss = nn.L1Loss()
        self.domain_num =  domain_num
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.relu = nn.ReLU()
        self. margin = margin


    def forward(self, input, D):
        posi, nega = torch.chunk(input, 2, dim=0)
        posi_D, nega_D = torch.chunk(D, 2, dim=0)

        #intra-domian
        posi_anchor = torch.softmax(torch_scatter.scatter_mean(posi, posi_D,dim_size=self.domain_num, dim=0),dim=-1)
        nega_anchor = torch.softmax(torch_scatter.scatter_mean(nega, nega_D,dim_size=self.domain_num, dim=0),dim=-1)
        posi = torch.softmax(posi, dim=-1)
        nega = torch.softmax(nega, dim=-1)

        if self.domain_num <=2:
            loss_c = (self.loss(posi.log(), posi_anchor[posi_D]) + self.loss(nega.log(), nega_anchor[nega_D]))

        else:
            idx_p = torch.fmod(posi_D + torch.randint(1, self.domain_num - 1, posi_D.size()).to(posi_D.device),
                               self.domain_num)
            idx_n = torch.fmod(nega_D + torch.randint(1, self.domain_num - 1, nega_D.size()).to(nega_D.device),
                               self.domain_num)
            loss_c = self.relu(self.margin + self.loss(posi.log(), posi_anchor[posi_D].detach()) - self.loss(posi.log(),posi_anchor[idx_p].detach())) + \
                     self.relu(self.margin + self.loss(nega.log(), nega_anchor[nega_D].detach()) - self.loss(nega.log(),nega_anchor[idx_n].detach()))

        # inter-domian
        posi_anchor_mean = torch.repeat_interleave(posi_anchor.mean(dim=0, keepdim =True), posi_anchor.size(0), dim=0)
        nega_anchor_mean = torch.repeat_interleave(nega_anchor.mean(dim=0, keepdim=True), nega_anchor.size(0), dim=0)

        loss_d = self.relu(self.margin + self.loss(posi_anchor.log(), posi_anchor_mean.detach()) + self.loss(nega_anchor.log(), nega_anchor_mean.detach())
                           - (self.loss(posi_anchor_mean.log(), nega_anchor_mean) +self.loss(nega_anchor_mean.log(), posi_anchor_mean))/2)

        return (loss_c +loss_d)/2

class Reconstruction_Loss(nn.Module):
    def __init__(self, margin=0.5) -> None:
        super().__init__()
        self.margin = margin
        # self.similarity = nn.CosineSimilarity()
        self.similarity = nn.L1Loss()

    def forward(self, x_rec, x):

        (batch, idx) = torch.nonzero(x.sum(-1), as_tuple=True)
        x = x[batch, idx, :]
        x_rec = x_rec[batch,idx ,:]
        dis = self.similarity(x_rec,x)

        return torch.mean(dis ,dim=0) + self.margin

class loss_function(nn.Module):
    def __init__(self, margin, domain_num):
        super(loss_function, self).__init__()
        self.mc_loss = Multilevel_Contrastive_Loss(margin=margin, domain_num= domain_num)
        self.kl = nn.KLDivLoss(reduction='mean')
        self.relu = nn.ReLU()
        self.reconstruct = Reconstruction_Loss(margin=margin)
        self.margin = margin

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], original= True):
        stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec,cont, sty = preds
        stylabels, y, x,  D = targets

        #loss for rumor detection
        clsloss_sty = F.cross_entropy(pred_sty +1e-6,y)
        clsloss_cont = F.cross_entropy(pred_cont +1e-6,y)

        # reconstruction loss
        loss_rec = self.reconstruct(x_rec, x)

        stylabels_nonzero = torch.softmax(stylabels - torch.max(stylabels, dim=-1, keepdim=True)[0], dim=-1)
        stypred_cont_nonzero = torch.softmax(stypred_cont - torch.max(stypred_cont, dim=-1, keepdim=True)[0], dim=-1)
        stypred_sty_nonzero = torch.softmax(stypred_sty - torch.max(stypred_sty, dim=-1, keepdim=True)[0],dim=-1)

        if original:
            styloss_sty = self.mc_loss(stypred_sty, D)
        else:
            styloss_sty = 0

        styloss_cont = self.kl(stypred_cont_nonzero.log(), stylabels_nonzero)
        loss = styloss_cont + styloss_sty + (clsloss_sty + clsloss_cont) * 0.5 + loss_rec


        return loss, styloss_cont, styloss_sty, clsloss_cont, clsloss_sty, loss_rec


class loss_function_MLP(nn.Module):
    def __init__(self, margin, domain_num):
        super(loss_function_MLP, self).__init__()
        self.relu = nn.ReLU()
        self.reconstruct = Reconstruction_Loss(margin=margin)
        self.margin = margin
        self.triple_loss = Triple_Loss(margin=margin, domain_num = domain_num)
        self.kl_loss = nn.KLDivLoss(reduction='mean')

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], original= True):
        pred_sty,pred_cont, cont_rec, sty_rec, stypred_sty = preds
        y, x, D = targets

        #loss for rumor detection
        clsloss_sty = F.cross_entropy(pred_sty +1e-6,y)
        clsloss_sty_mix = F.cross_entropy(stypred_sty+1e-6, y)
        clsloss_cont = F.cross_entropy(pred_cont +1e-6,y)

        #reconstruction loss
        loss_rec_cont = self.reconstruct(cont_rec,x)
        loss_rec_sty = self.reconstruct(sty_rec,x)
        # loss_rec = 0

        if original:
            loss_sty = self.triple_loss(stypred_sty, D)
            loss = clsloss_sty + clsloss_cont + (loss_rec_cont + loss_rec_sty) * 0.5 + clsloss_sty_mix + loss_sty
        else:
            loss = clsloss_sty + clsloss_cont + (loss_rec_cont + loss_rec_sty) * 0.5 + clsloss_sty_mix

        # if original:
        #
        #     loss = clsloss_sty + clsloss_cont + (loss_rec_cont + loss_rec_sty) * 0.5 + loss_sty
        # else:
        #     loss_sty = self.kl_loss(stypred_sty.log(), stypred_label)
        #     loss = clsloss_sty + clsloss_cont + (loss_rec_cont + loss_rec_sty) * 0.5 + loss_sty
        return loss, clsloss_sty, loss_rec_cont, loss_rec_sty

class loss_function_wogrl(nn.Module):
    def __init__(self, margin,domain_num):
        super(loss_function_wogrl, self).__init__()
        self.triple_loss = Multilevel_Contrastive_Loss(margin=margin,domain_num=domain_num)
        self.domain_num = domain_num
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.reconstruct = Reconstruction_Loss(margin=margin)

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], original= True):
        stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec,cont, sty = preds
        stylabels, y, x, D = targets

        #loss for rumor detection
        clsloss_sty = F.cross_entropy(pred_sty +1e-6,y)
        clsloss_cont = F.cross_entropy(pred_cont+1e-6,y)

        #reconstruction loss
        loss_rec = self.reconstruct(x_rec,x)
        # loss_rec = 0

       # avoiding the influence of padding data
        stylabels_nonzero = torch.softmax(stylabels - torch.max(stylabels, dim=-1, keepdim=True)[0], dim=-1)
        # stypred_cont_nonzero = torch.softmax(stypred_cont - torch.max(stypred_cont, dim=-1, keepdim=True)[0], dim=-1)
        stypred_sty_nonzero = torch.softmax(stypred_sty - torch.max(stypred_sty, dim=-1, keepdim=True)[0], dim=-1)

        if  original:
            styloss_sty = self.triple_loss(stypred_sty, D)
        else:
            styloss_sty = 0
        styloss_cont = 0
        loss = styloss_cont + styloss_sty + (clsloss_sty + clsloss_cont) * 0.5 + loss_rec

        return loss, styloss_cont, styloss_sty, clsloss_cont, clsloss_sty, loss_rec

class loss_function_wostyledetection(nn.Module):
    def __init__(self, margin, domain_num):
        super(loss_function_wostyledetection, self).__init__()
        self.domain_num= domain_num
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.reconstruct = Reconstruction_Loss(margin=margin)

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], original= True):
        stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec,cont, sty = preds
        stylabels, y, x, D = targets

        #loss for rumor detection
        clsloss_sty = F.cross_entropy(pred_sty +1e-6,y)
        clsloss_cont = F.cross_entropy(pred_cont+1e-6,y)

        #reconstruction loss
        loss_rec = self.reconstruct(x_rec,x)

        styloss_sty = 0
        styloss_cont = 0  # 不参与计算
        loss = styloss_cont + styloss_sty + (clsloss_sty + clsloss_cont) * 0.5 + loss_rec

        return loss, styloss_cont, styloss_sty, clsloss_cont, clsloss_sty, loss_rec

class loss_function_wodecoder(nn.Module):
    def __init__(self, margin, domain_num):
        super(loss_function_wodecoder, self).__init__()
        self.triple_loss = Multilevel_Contrastive_Loss(margin=margin, domain_num= domain_num)
        self.kl = nn.KLDivLoss(reduction='mean')
        self.relu = nn.ReLU()
        self.reconstruct = Reconstruction_Loss(margin=margin)
        self.margin = margin

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], original= True):
        stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec,cont, sty = preds
        stylabels, y, x,  D = targets

        #loss for rumor detection
        clsloss_sty = F.cross_entropy(pred_sty +1e-6,y)
        clsloss_cont = F.cross_entropy(pred_cont +1e-6,y)

        # reconstruction loss
        loss_rec = 0

        stylabels_nonzero = torch.softmax(stylabels - torch.max(stylabels, dim=-1, keepdim=True)[0], dim=-1)
        stypred_cont_nonzero = torch.softmax(stypred_cont - torch.max(stypred_cont, dim=-1, keepdim=True)[0], dim=-1)
        stypred_sty_nonzero = torch.softmax(stypred_sty - torch.max(stypred_sty, dim=-1, keepdim=True)[0],dim=-1)

        if  original:
            styloss_sty = self.triple_loss(stypred_sty, D)
        else:
            styloss_sty = 0
        styloss_cont = self.kl(stypred_cont_nonzero.log(), stylabels_nonzero)
        loss = styloss_cont + styloss_sty + (clsloss_sty + clsloss_cont) * 0.5 + loss_rec

        return loss, styloss_cont, styloss_sty, clsloss_cont, clsloss_sty, loss_rec


class loss_function_tripleLoss(nn.Module):
    def __init__(self, margin,domain_num):
        super(loss_function_tripleLoss, self).__init__()
        self.triple_loss = Triple_Loss(margin=margin,domain_num=domain_num)
        self.domain_num = domain_num
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.reconstruct = Reconstruction_Loss(margin=margin)

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], original= True):
        stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec,cont, sty = preds
        stylabels, y, x, D = targets

        #loss for rumor detection
        clsloss_sty = F.cross_entropy(pred_sty +1e-6,y)
        clsloss_cont = F.cross_entropy(pred_cont+1e-6,y)

        #reconstruction loss
        loss_rec = self.reconstruct(x_rec,x)
        # loss_rec = 0

       # avoiding the influence of padding data
        stylabels_nonzero = torch.softmax(stylabels - torch.max(stylabels, dim=-1, keepdim=True)[0], dim=-1)
        stypred_cont_nonzero = torch.softmax(stypred_cont - torch.max(stypred_cont, dim=-1, keepdim=True)[0], dim=-1)
        stypred_sty_nonzero = torch.softmax(stypred_sty - torch.max(stypred_sty, dim=-1, keepdim=True)[0], dim=-1)

        if  original:
            styloss_sty = self.triple_loss(stypred_sty, D)
        else:
            styloss_sty = 0
        styloss_cont = self.kl(stypred_cont_nonzero.log(), stylabels_nonzero)
        loss = styloss_cont + styloss_sty + (clsloss_sty + clsloss_cont) * 0.5 + loss_rec

        return loss, styloss_cont, styloss_sty, clsloss_cont, clsloss_sty, loss_rec


class loss_function_QuadrupleLoss(nn.Module):
    def __init__(self, margin, domain_num):
        super(loss_function_QuadrupleLoss, self).__init__()
        self.triple_loss = Quadruple_Loss(margin=margin, domain_num=domain_num)
        self.domain_num = domain_num
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.reconstruct = Reconstruction_Loss(margin=margin)

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], original=True):
        stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec, cont, sty = preds
        stylabels, y, x, D = targets

        # loss for rumor detection
        clsloss_sty = F.cross_entropy(pred_sty + 1e-6, y)
        clsloss_cont = F.cross_entropy(pred_cont + 1e-6, y)

        # reconstruction loss
        loss_rec = self.reconstruct(x_rec, x)
        # loss_rec = 0

        # avoiding the influence of padding data
        stylabels_nonzero = torch.softmax(stylabels - torch.max(stylabels, dim=-1, keepdim=True)[0], dim=-1)
        stypred_cont_nonzero = torch.softmax(stypred_cont - torch.max(stypred_cont, dim=-1, keepdim=True)[0], dim=-1)
        stypred_sty_nonzero = torch.softmax(stypred_sty - torch.max(stypred_sty, dim=-1, keepdim=True)[0], dim=-1)

        if  original:
            styloss_sty = self.triple_loss(stypred_sty, D)
        else:
            styloss_sty = 0
        styloss_cont = self.kl(stypred_cont_nonzero.log(), stylabels_nonzero)
        loss = styloss_cont + styloss_sty + (clsloss_sty + clsloss_cont) * 0.5 + loss_rec

        return loss, styloss_cont, styloss_sty, clsloss_cont, clsloss_sty, loss_rec


class loss_function_DomainLoss(nn.Module):
    def __init__(self, margin,domain_num):
        super(loss_function_DomainLoss, self).__init__()
        self.domain_num = domain_num
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.reconstruct = Reconstruction_Loss(margin=margin)

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], original= True):
        stypred_cont, stypred_sty, pred_cont, pred_sty, x_rec,cont, sty = preds
        stylabels, y, x, D = targets

        #loss for rumor detection
        clsloss_sty = F.cross_entropy(pred_sty +1e-6,y)
        clsloss_cont = F.cross_entropy(pred_cont+1e-6,y)

        #reconstruction loss
        loss_rec = self.reconstruct(x_rec,x)
        # loss_rec = 0

       # avoiding the influence of padding data

        if  original:
            styloss_sty = F.cross_entropy(stypred_sty, D)
        else:
            styloss_sty = 0
        styloss_cont =0

        loss = (clsloss_sty +clsloss_cont)*0.5 +styloss_sty +styloss_cont + loss_rec

        return loss, styloss_cont, styloss_sty, clsloss_cont, clsloss_sty, loss_rec
