# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/10/19 9:31
@Author     : Danke Wu
@File       : utils.py
"""
import torch
from torch.autograd import Variable, Function
import torch.nn as nn
import torch.nn.functional as F
from typing import *

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        if 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)<=1:
            return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
        else:
            return 1.0 - (self.n_epochs-1)/self.n_epochs

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


# def MIXUP( invariant, variant, x):
#     # fake style data generation
#
#     x_non, x_rumor = torch.chunk(x, 2, dim=0)
#     z_inv_non, z_inv_rumor = torch.chunk(invariant, 2, dim=0)
#     z_var_non, z_var_rumor = torch.chunk(variant, 2, dim=0)
#     index_non = torch.randperm(z_var_non.size()[0])
#     index_rumor = torch.randperm(z_var_rumor.size()[0])
#     mix_var_non = z_var_non[index_non]
#     mix_var_rumor = z_var_rumor[index_rumor]
#
#     #addition
#     xr_fake = (torch.mul(z_inv_rumor, (mix_var_rumor!=0).float()) + mix_var_rumor)
#     xn_fake = (torch.mul(z_inv_non, (mix_var_non!=0).float() ) + mix_var_non)
#
#     # # multiple
#     # xr_fake = torch.mul(mix_var_rumor, z_inv_rumor)
#     # xn_fake = torch.mul(mix_var_non, z_inv_non)
#
#     # concatenate
#     # xr_fake = torch.cat((z_inv_rumor, mix_var_rumor),dim=-1)
#     # xn_fake = torch.cat((z_inv_non, mix_var_non),dim=-1)
#
#     mix_index = torch.cat((index_non, index_rumor), dim=0)
#     x_mix = torch.cat((xn_fake, xr_fake), dim=0)
#     x_reidx = torch.cat((x_non[index_non], x_rumor[index_rumor]),dim=0)
#     mask_nonzero = torch.nonzero(x_reidx.sum(-1), as_tuple=True)
#
#     return x_mix, x_reidx, mask_nonzero


def MIXUP( invariant, variant, x):
    # fake style data generation

    x_non, x_rumor = torch.chunk(x, 2, dim=0)
    z_inv_non, z_inv_rumor = torch.chunk(invariant, 2, dim=0)
    z_var_non, z_var_rumor = torch.chunk(variant, 2, dim=0)
    index_non = torch.randperm(z_var_non.size()[0])
    index_rumor = torch.randperm(z_var_rumor.size()[0])
    mix_var_non = z_var_non[index_non]
    mix_var_rumor = z_var_rumor[index_rumor]

    #addition
    xr_fake = (torch.mul(z_inv_rumor, (mix_var_rumor!=0).float()) + mix_var_rumor)
    xn_fake = (torch.mul(z_inv_non, (mix_var_non!=0).float() ) + mix_var_non)

    mix_index = torch.cat((index_non, index_rumor), dim=0)
    x_mix = torch.cat((xn_fake, xr_fake), dim=0)
    x_reidx = torch.cat((x_non[index_non], x_rumor[index_rumor]),dim=0)
    mask_nonzero = torch.nonzero(x_reidx.sum(-1), as_tuple=True)

    return x_mix, x_reidx, mask_nonzero

def MIXUP_Y( invariant, variant, x, y):
    # fake style data generation

    idx_non, idx_rumor = torch.where(y==0), torch.where(y==1)
    x_non, x_rumor = x[idx_non],x[idx_rumor]
    z_inv_non, z_inv_rumor = invariant[idx_non],invariant[idx_rumor]
    z_var_non, z_var_rumor = variant[idx_non],variant[idx_rumor]
    index_non = torch.randperm(idx_non.size()[0])
    index_rumor = torch.randperm(idx_rumor.size()[0])

    #addition
    xr_fake = (torch.mul(z_inv_rumor, (z_var_rumor[index_rumor]!=0).float()) + z_var_rumor[index_rumor])
    xn_fake = (torch.mul(z_inv_non, (z_var_non[index_non]!=0).float() ) + z_var_non[index_non])

    index = torch.cat((index_non, index_rumor), dim=0)
    _, index_ori = torch.sort(index)
    x_reidx = x[index_ori]
    x_mix = torch.cat((xn_fake, xr_fake), dim=0)
    x_reidx = torch.cat((x_non[index_non], x_rumor[index_rumor]),dim=0)
    mask_nonzero = torch.nonzero(x_reidx.sum(-1), as_tuple=True)

    return x_mix, x_reidx, mask_nonzero

def MIXUP_cont( invariant, variant):
    # fake style data generation
    z_inv_non, z_inv_rumor = torch.chunk(invariant, 2, dim=0)
    z_var_non, z_var_rumor = torch.chunk(variant, 2, dim=0)
    index_non = torch.randperm(z_var_non.size()[0])
    index_rumor = torch.randperm(z_var_rumor.size()[0])
    mix_var_non = z_var_non[index_non]
    mix_var_rumor = z_var_rumor[index_rumor]

    #addition
    # xr_fake = mix_var_rumor * (torch.mul(mix_var_rumor, z_inv_rumor) != 0).float() + z_inv_rumor * (
    #             torch.mul(z_inv_rumor, mix_var_rumor) != 0).float()
    # xn_fake = mix_var_non * (torch.mul(mix_var_non, z_inv_non) != 0).float() + z_inv_non * (
    #             torch.mul(z_inv_non, mix_var_non) != 0).float()
    xr_fake = mix_var_rumor + z_inv_rumor
    xn_fake = mix_var_non + z_inv_non

    x_mix = torch.cat((xn_fake, xr_fake), dim=0)
    reidx = torch.cat((index_non, index_rumor + index_non.size(0) ), dim=0)

    return x_mix, reidx

def MIXUP_sty( style):
    # fake style data generation
    style_non, style_rumor = torch.chunk(style, 2, dim=0)
    for d in range(style_non.size(-1)):
        idx = torch.randperm(style_non.size(0))
        style_non[:,d] = style_non[idx,d]
    for d in range(style_rumor.size(-1)):
        idx = torch.randperm(style_rumor.size(0))
        style_rumor[:, d] = style_rumor[idx, d]
    style_mix = torch.cat((style_non, style_rumor), dim=0)
    return style_mix

def MIXUP_sentlevel( invariant, variant, x):
    # fake style data generation
    x_non, x_rumor = torch.chunk(x, 2, dim=0)
    z_inv_non, z_inv_rumor = torch.chunk(invariant, 2, dim=0)
    z_var_non, z_var_rumor = torch.chunk(variant, 2, dim=0)
    index_non = torch.randperm(z_var_non.size()[0])
    index_rumor = torch.randperm(z_var_rumor.size()[0])
    mix_var_non = z_var_non[index_non]
    mix_var_rumor = z_var_rumor[index_rumor]

    #addition
    xr_fake = (torch.mul(z_inv_rumor, (mix_var_rumor!=0).float()) + mix_var_rumor)
    xn_fake = (torch.mul(z_inv_non, (mix_var_non!=0).float() ) + mix_var_non)

    # # multiple
    # xr_fake = torch.mul(mix_var_rumor, z_inv_rumor)
    # xn_fake = torch.mul(mix_var_non, z_inv_non)

    # concatenate
    # xr_fake = torch.cat((z_inv_rumor, mix_var_rumor),dim=-1)
    # xn_fake = torch.cat((z_inv_non, mix_var_non),dim=-1)

    mix_index = torch.cat((index_non, index_rumor), dim=0)
    x_mix = torch.cat((xn_fake, xr_fake), dim=0)
    x_reidx = torch.cat((x_non[index_non], x_rumor[index_rumor]),dim=0)
    mask_nonzero = torch.nonzero(x_reidx.sum(-1), as_tuple=True)

    return x_mix, x_reidx, mask_nonzero

class Mine(nn.Module):
    def __init__(self,f,s):
        super().__init__()
        self.fc1_x = nn.Linear(f, 50, bias = False)
        self.fc1_y = nn.Linear(s, 50, bias = False)
        self.fc2 = nn.Linear(50, 1, bias = False)

    def forward(self, x,y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 =self.fc2(h1)
        return h2

    def mutual_est(self,x, y, mask):
        #remove the padding data, shuffile the sample within class
        cls1, = torch.where(mask < x.size()[0]/2)
        cls2, = torch.where(mask >= x.size()[0]/2)
        y1, y2 = y[mask[cls1]], y[mask[cls2]]
        shuffile_idx1 = torch.randperm(cls1.size()[0])
        shuffile_idx2 = torch.randperm(cls2.size()[0])
        y_ = torch.cat((y1[shuffile_idx1],y2[shuffile_idx2]),dim=0)

        x,y, y_ = x[mask], y[mask], y_
        joint, marginal = self(x, y), self(x, y_)
        #maximize loss equals to minimize -loss
        loss = torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))
        return F.relu(2 - loss)

class Mine_sent(nn.Module):
    def __init__(self,f,s):
        super().__init__()
        self.fc1_x = nn.Linear(f, 50, bias = False)
        self.fc1_y = nn.Linear(s, 50, bias = False)
        self.fc2 = nn.Linear(50, 1, bias = False)

    def forward(self, x,y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2

    def mutual_est(self, x, y):

        y1, y2 = torch.chunk(y,2,dim=0)
        shuffile_idx1 = torch.randperm(y1.size()[0])
        shuffile_idx2 = torch.randperm(y2.size()[0])
        y_ = torch.cat((y1[shuffile_idx1],y2[shuffile_idx2]),dim=0)

        joint, marginal = self(x, y), self(x, y_)
        #maximize loss equals to minimize -loss
        loss = torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))
        return F.relu(2 - loss)

