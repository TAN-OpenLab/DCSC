# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/6/17 11:31
@Author     : Danke Wu
@File       : model_layers.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math

class sentence_embedding(nn.Module):
    def __init__(self,h_in, h_out):
        super(sentence_embedding, self).__init__()
        self.embedding = nn.Linear(h_in, h_out, bias=False)
        self.leakrelu = nn.LeakyReLU()

    def forward(self,x):

        x = self.embedding(x)
        x = self.leakrelu(x)
        return x


def hyper_Adjentmatrix(adj,node_mask):

    adj_T = adj.permute(0, 2, 1)  # bs x c x 1 x n x n
    adj_T_2 = torch.matmul(adj_T, adj_T)
    adj_T_3 = torch.matmul(adj_T_2, adj_T)
    adj_agg = adj_T.detach()
    adj_agg.data.masked_fill_(adj_T_2 != 0, 1)
    adj_agg.data.masked_fill_(adj_T_3 != 0, 1)
    d = adj_agg.sum(dim=-1, keepdim=True).detach()
    d = torch.where(d != 0, torch.pow(d, -1), d)
    adj_agg = torch.mul(adj_agg, d)
    adj_agg[:, 0, 0] = 1
    adj_agg.data.masked_fill_(node_mask, 0)
    return adj_agg


class Context_Aggragation(nn.Module):
    def __init__(self, f_in):
        super(Context_Aggragation, self).__init__()

        self.weight = nn.Linear(2*f_in, 1, bias= False)

    def forward(self, h, adj):
        B, N, _ = h.size()
        #print(torch.sum(h[0, -1, :, :], dim=-1) == 0)
        node_mask = (torch.sum(h,dim=-1,keepdim=True) ==0).repeat(1,1,N)
        adj_agg = hyper_Adjentmatrix(adj, node_mask)

        h_context = torch.matmul(adj_agg, h)
        h_agg = torch.cat( (h, h_context),dim=-1)
        attention = self.weight(h_agg)
        s_agg = torch.mul(h,attention) + torch.mul(h_context, 1-attention)
        return s_agg


class Stance_Extraction(nn.Module):
    def __init__(self, f_in):
        super(Stance_Extraction,self).__init__()
        # self.Q = nn.Linear(f_in, f_in, bias=False)
        # self.K = nn.Linear(f_in, f_in, bias=False)
        self.stance_layer = nn.Sequential(nn.Linear(f_in *2+1, f_in, bias= False),
                                          nn.LeakyReLU(),
                                          nn.Linear(f_in, 1, bias= False),
                                          nn.Tanh())
        # self.op = nn.Parameter(torch.Tensor(f_in, 1))


        # nn.init.xavier_uniform_(self.op)

    def forward(self,h):
        B, N, _= h.size()

        #基于距离的相似度
        # h_dis = torch.exp(-torch.abs(h_src - h))
        # h_dis = torch.matmul(h_dis,self.op)
        # h_lab = torch.cosine_similarity(h,h_src,dim=-1).unsqueeze(dim=-1)
        # h_lab = torch.where(h_lab>=0, 1.0, -1.0)
        # attn_op = torch.mul(h_dis,h_lab)

        # #基于欧氏距离相似度
        # h_dis = h_src - h
        # h_dis = torch.matmul(h_dis, self.op)
        # attn_op = self.tanh(h_dis)

        # #cosine 相似度
        # h_dis = torch.cosine_similarity(h_src, h, dim=-1).unsqueeze(dim=-1)
        # h_dis = torch.mul(h_dis, self.op)
        # attn_op = self.tanh(h_dis)

        # #基于FC的相似度
        # h_src = torch.repeat_interleave(h[:,0,:].unsqueeze(dim=1), N, dim=1)
        mask = (h !=0).float()
        h_src = torch.repeat_interleave(h[:,0,:].unsqueeze(dim=1), N, dim=1)
        h_src = torch.mul(h_src, mask.detach())
        h_multiple = torch.mul(h_src, h)
        h_dis_L1 = torch.abs(h_src - h)
        h_cos_sim = torch.cosine_similarity( h_src, h, dim=-1).unsqueeze(dim=-1)
        h_comb = torch.cat((h_multiple,h_dis_L1,h_cos_sim), dim=-1)
        stance = self.stance_layer(h_comb)


        # # 基于拼接的FC的相似度
        # h_dis = torch.cat((h_src, h),dim=-1)
        # attn_op = torch.matmul(h_dis, self.op)
        # #
        # attn_op = self.tanh(attn_op)
        # attn_op = self.dropout(attn_op)

        return stance

class BatchGAT(nn.Module):
    def __init__(self, n_heads, f_in, h_hids, dropout, attn_dropout, bias =False):
        super(BatchGAT, self).__init__()

        self.n_layer = len(h_hids)
        self.f_in = f_in
        self.dropout = dropout
        self.bias = bias

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            f_in = h_hids[i-1] * n_heads[i-1] if i else self.f_in
            self.layer_stack.append(
                BatchMultiHeadGraphAttention( n_heads[i], f_in=f_in,
                                             f_out=h_hids[i], attn_dropout=attn_dropout,bias =self.bias)
            )

    def forward(self, x, adj, mask):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj, mask) # bs x c x n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)

            else:
                x = F.elu(x.permute(0, 2, 1, 3).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return x


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias = False):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.fc = nn.Linear(f_in, f_out, bias= False)
        self.attn_src_linear = nn.Linear(f_out, 1 * n_head, bias= False )
        self.attn_dst_linear = nn.Linear(f_out, 1 * n_head, bias= False)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)


    def init_glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def forward(self, h, adj, mask):
        (batch, row) = mask
        # node_mask = h.sum(-1,keepdim=True) == 0
        # h = h.unsqueeze(1)
        h_prime = self.fc(h)
        # h_prime = torch.matmul(h, self.w) # bs x c x n_head x n x f_out
        B, N, _ = h.size()  # h is of size bs x c x n x f_in


        attn_src = self.attn_src_linear(h_prime).permute(0, 2, 1).unsqueeze(dim=-1)
        attn_dst = self.attn_dst_linear(h_prime).permute(0, 2, 1).unsqueeze(dim=-1)

        attn = attn_src.expand(-1, -1,-1, N) + attn_dst.expand(-1, -1,-1, N).permute(0, 1, 3, 2) # bs  x c x n_head x n x n
        attn_all = self.leaky_relu(attn)

        adj[batch, row, row] = 1
        adj = torch.repeat_interleave(adj.unsqueeze(dim=1), self.n_head, dim=1)
        attn_all.masked_fill_(adj == 0, 0.0)
        attn_mask = attn_all[batch, :, row, :]
        attn_mask.masked_fill_(attn_mask == 0., -1e20)
        attn_mask = self.softmax(attn_mask)
        attn_all[batch, :, row, :] = attn_mask
        attn_all = self.dropout(attn_all)
        # adj_ = torch.where(adj == 1, 0.0, -1.0e12).detach()
        # attn_all = self.softmax(attn_all + adj_)

        h_prime = h_prime.unsqueeze(1)
        output = torch.matmul(attn_all, h_prime) # bs x c x n_head x n x f_out

        if self.bias is not None:
            return output + self.bias
        else:
            return output



class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p =1, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)
        mask_zero = x==0.0

        # mu = x.mean(dim=[1, 2], keepdim=True)
        # var = x.var(dim=[1, 2], keepdim=True)
        x_root = x[:,1,:].unsqueeze(dim=1)
        mu = x_root.mean(dim=2, keepdim=True)
        var = x_root.var(dim=2, keepdim=True)
        # mu = mu.sum(dim= 1, keepdim=True) / ((mu != 0).sum(dim=1, keepdim=True))
        # var = var.sum(dim=1, keepdim=True) / ((var != 0).sum(dim=1, keepdim=True))
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossclass':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        # mu_mix = mu * lmda + mu2 * (1-lmda)
        # sig_mix = sig * lmda + sig2 * (1-lmda)
        # x_aug = x_normed * sig_mix + mu_mix
        x_aug = x_normed * sig2 + mu2
        x_aug.masked_fill_(mask_zero, 0.0)

        return x_aug

# def Stance_Dropout(x, num_post):
#     stance_num = torch.count_nonzero(x.detach(), dim=1)
#     rand_drop = [random.randint(1,x) for x in stance_num ]
#     rand_drop = torch.tensor(rand_drop, device= x.device).unsqueeze(dim=1)
#     mask = x!=0
#
#
#     scale = (stance_num/rand_drop).unsqueeze(dim=1).repeat_interleave(num_post, dim=1)
#     x = torch.mul(scale,x)
#     return x




class StR_GRD(nn.Module):
    def __init__(self, c_in, c_hid, num_posts):
        super( StR_GRD, self).__init__()

        self.embedding = sentence_embedding(c_in, c_hid)
        self.cont_agg = Context_Aggragation(c_hid)
        self.stance_extraction= Stance_Extraction(c_hid)
        self.gat = BatchGAT([1,1], 1, [10,1], 0.0, 0.0) #n_heads, f_in, h_hids, dropout, attn_dropout,
        self.classifier = nn.Linear(num_posts *2, 2)
        self.num_posts = num_posts


    def forward(self, x, A):

        #student
        x_emb = self.embedding(x)
        x_contxt = self.cont_agg(x_emb,A)
        x_stance = self.stance_extraction(x_contxt)
        mask = torch.nonzero(x_stance.squeeze(dim=-1), as_tuple=True)
        x_stance_gat = self.gat(x_stance, A, mask)
        h_mix = torch.cat((x_stance.squeeze(dim=-1), x_stance_gat.squeeze(dim=-1)),dim=-1)
        preds = self.classifier(h_mix)

        return preds
