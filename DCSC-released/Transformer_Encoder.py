# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/6/26 14:55
@Author     : Danke Wu
@File       : Transformer_Encoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttn_Root_Enhence(nn.Module):
    def __init__(self,  n_head, h_in, dropout=0.1, attn_scale=False, root_enhance= True):
        """

        :param h_in:
        :param n_head:
        :param scale: 是否scale输出
        """
        super().__init__()
        assert h_in % n_head==0

        self.n_head = n_head

        self.root_enhance = root_enhance
        if root_enhance:
            self.q_linear = nn.Linear(h_in,  h_in, bias=False)
            self.kv_linear = nn.Linear(2 * h_in, 2 * h_in, bias=False)
        else:
            self.qkv_linear = nn.Linear(h_in, 3* h_in, bias=False)

        self.dropout_layer = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU()

        if attn_scale:
            self.scale = math.sqrt(h_in//n_head)
        else:
            self.scale = 1


    def forward(self, x, mask_nonzero):
        """

        :param x: bsz x max_len x h_in
        :param mask: bsz x max_len
        :return:
        """
        batch_size, max_len, h_in = x.size()
        if self.root_enhance:

            #index_nonzeromask
            (batch, row) = mask_nonzero
            root = torch.zeros_like(x,device=x.device)
            root[batch,row,:] = x[batch,0,:]

            x_enhance = torch.cat((x, root), dim=-1)
            q = self.q_linear(x)
            kv = self.kv_linear(x_enhance)
            k, v = torch.chunk(kv, 2, dim=-1)
        else:
            qkv = self.qkv_linear(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
        attn = attn/self.scale

        #index_nonzoremask
        (batch, row) = mask_nonzero
        attn_mask = attn[batch, :, row,:]
        attn_mask.masked_fill_(attn_mask==0., -1e20)
        attn_mask = F.softmax(attn_mask, dim=-1)
        attn[batch, :, row,:] = attn_mask
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x h_in//n_head
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.leakyrelu(v)


        return v


class TransformerLayer(nn.Module):
    def __init__(self, n_head, h_in, h_hid, dropout, after_norm, dropout_attn, attn_scale, root_enhance):
        """

        :param int h_in: 一般512之类的
        :param self_attn: self attention模块，输入为x:batch_size x max_len x h_in, mask:batch_size x max_len, 输出为
            batch_size x max_len x h_in
        :param int h_hid: FFN中间层的dimension的大小
        :param bool after_norm: norm的位置不一样，如果为False，则embedding可以直接连到输出
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(h_in)
        self.norm2 = nn.LayerNorm(h_in)

        self.self_attn = MultiHeadAttn_Root_Enhence(n_head, h_in, dropout_attn, attn_scale, root_enhance)

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(h_in, h_hid),
                                 nn.LeakyReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(h_hid, h_in))



    def forward(self, x, mask_nonzero):
        """

        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)


        x = self.self_attn(x, mask_nonzero)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)

        residual = x
        x = self.ffn(x)
        x = x + residual
        if self.after_norm:
            x = self.norm2(x)

        mask_nonzero_matrix = torch.clone(x)
        (batch, row) = mask_nonzero
        mask_nonzero_matrix[batch, row, :] = torch.zeros_like(mask_nonzero_matrix[0, 0, :])
        x = x - mask_nonzero_matrix.detach()

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, n_head, h_in, h_hid, dropout, after_norm= True, dropout_attn=None, attn_scale=False, root_enhance = True ):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.h_in = h_in


        self.layers = nn.ModuleList([TransformerLayer(n_head, h_in, h_hid, dropout, after_norm, dropout_attn, attn_scale, root_enhance)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        """

        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """

        for layer in self.layers:
            x = layer(x, mask)
        return x



class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, n_head, h_in, h_hid, dropout, after_norm=True, dropout_attn= None, attn_scale=False, root_enhance=False):
        super(TransformerDecoder,self).__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.h_in = h_in

        self.layers = nn.ModuleList([TransformerLayer( n_head, h_in,  h_hid, dropout, after_norm, dropout_attn, attn_scale, root_enhance)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        """

        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """

        for layer in self.layers:
            x = layer(x, mask)
        return x
