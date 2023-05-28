# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from Tools.WHA_ysy import WHA_ysy
from utils.config import DefaultConfig
from Tools.WMSA import WMSA

configs = DefaultConfig()

class RAGru(nn.Module):

    def __init__(self,
                 vocab_size,
                 n_class,
                 embed_dim=49,
                 rnn_hidden=256,
                 ifmha=False,
                 mha_in_features=49,
                 ):
        super(RAGru, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.gru = nn.GRU(input_size=embed_dim,
                          hidden_size=rnn_hidden,
                          batch_first=True)

        self.fc = nn.Linear(in_features=rnn_hidden,
                            out_features=n_class)

        self.wha_ysy = WHA_ysy(in_features=mha_in_features, head_num=1, weight_dim0=configs.batch_size, weight_dim1=1)

        self.ifmha = ifmha
        self.wmsa = WMSA(channels=configs.batch_size)
        self.wmsa2 = WMSA(channels=configs.batch_size)

        self.emb = nn.Embedding(num_embeddings=500*49,embedding_dim=684)
        self.gru2 = nn.GRU(input_size=embed_dim,
                          hidden_size=rnn_hidden,
                          batch_first=True)

    def forward(self, x):

        x_ori = x
        if self.ifmha:
            x = self.wha_ysy(x, x, x)
            x = x * x_ori
        output1, h_n_1 = self.gru(x)

        x = h_n_1.squeeze()

        x = self.fc(x)
        x = torch.sigmoid(x)

        features_g = self.wmsa(output1.reshape(500, configs.batch_size, 1, 49),
                              x_ori.reshape(500, configs.batch_size, 1, 49))
        features_g = features_g.reshape(configs.batch_size, 500, 49)

        return features_g, x

