# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/14 9:41
@Author     : Danke Wu
@File       : EANN.py
"""
# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/5/12 10:08
@Author     : Danke Wu
@File       : EANN.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F
from Transformer_Encoder import TransformerEncoder
from utils import GRL

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


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, domain_num, filter_num, f_in, emb_dim,  dropout):
        super(CNN_Fusion, self).__init__()

        self.domain_num = domain_num

        self.hidden_size = emb_dim
        self.lstm_size = emb_dim

        # self.embed = nn.Linear(f_in, emb_dim, bias=False)
        # # TEXT RNN
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### TEXT CNN
        channel_in = 1
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, f_in)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.grad_reversal = GRL(lambda_=1)

        ## Class  Classifier
        self.class_classifier = nn.Sequential(nn.Linear( self.hidden_size, 2),
                                              nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                               nn.LeakyReLU(True),
                                               nn.Linear(self.hidden_size, self.domain_num),
                                              nn.Softmax(dim=1))


    def forward(self, text,D):

        ##########CNN##################
        text = text.unsqueeze(dim=1)# add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.fc1(text))

        ### Fake or real
        class_output = self.class_classifier(text)
        ## Domain (which Event )
        reverse_feature = self.grad_reversal(text)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


# Neural Network Model (1 hidden layer)
class Transformer_Fusion(nn.Module):
    def __init__(self, domain_num, filter_num, f_in, emb_dim,  dropout):
        super(Transformer_Fusion, self).__init__()

        self.domain_num = domain_num

        self.hidden_size = emb_dim
        self.lstm_size = emb_dim

        # self.embed = nn.Linear(f_in, emb_dim, bias=False)
        # # TEXT RNN
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### transformer
        self.embedding = sentence_embedding(f_in, emb_dim)
        self.extractor = TransformerEncoder(1, 2, emb_dim,  emb_dim, dropout)
        self.grad_reversal = GRL(lambda_=1)

        ## Class  Classifier
        self.class_classifier = nn.Sequential(nn.Linear( self.hidden_size, 2),
                                              nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                               nn.LeakyReLU(True),
                                               nn.Linear(self.hidden_size, self.domain_num),
                                               nn.Softmax(dim=1))


    def forward(self, text, D):
        mask_nonzero = torch.nonzero(text.sum(-1), as_tuple=True)
        text = self.embedding(text, mask_nonzero)
        text = self.extractor(text, mask_nonzero)
        text = torch.mean(text, dim=1)

        ### Fake or real
        class_output = self.class_classifier(text)
        ## Domain (which Event )
        reverse_feature = self.grad_reversal(text)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output