# -*- encoding:utf8 -*-

import os
import random
import time
import sys

import torch
import torch as t
from torch import nn
import numpy as np

from mymodels import WMSA, RAGRU, ResAttn

from mlp_mixer_pytorch import MLPMixer

seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# from basic_module import BasicModule
from models.BasicModule import BasicModule

sys.path.append("../")
from utils.config import DefaultConfig

configs = DefaultConfig()


class ConvsLayer(BasicModule):
    def __init__(self, ):
        super(ConvsLayer, self).__init__()

        self.kernels = configs.kernels
        hidden_channels = configs.cnn_chanel
        in_channel = 1
        features_L = configs.max_sequence_length
        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        W_size = seq_dim + dssp_dim + pssm_dim

        padding1 = (self.kernels[0] - 1) // 2
        padding2 = (self.kernels[1] - 1) // 2
        padding3 = (self.kernels[2] - 1) // 2
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding1, 0),
                                        kernel_size=(self.kernels[0], W_size)))
        # self.conv1.add_module("BN1", nn.BatchNorm2d(228))
        self.conv1.add_module("ReLU", nn.PReLU())
        self.conv1.add_module("pooling1", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding2, 0),
                                        kernel_size=(self.kernels[1], W_size)))
        # self.conv2.add_module("BN2", nn.BatchNorm2d(228))
        self.conv2.add_module("ReLU", nn.ReLU())
        self.conv2.add_module("pooling2", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding3, 0),
                                        kernel_size=(self.kernels[2], W_size)))
        # self.conv3.add_module("BN3", nn.BatchNorm2d(228))
        self.conv3.add_module("ReLU", nn.ReLU())
        self.conv3.add_module("pooling3", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

    def forward(self, x):
        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)  # features3.shape torch.Size([32, 228, 1, 1])

        features = t.cat((features1, features2, features3), 1)  # features.shape torch.Size([32, 684, 1, 1])

        shapes = features.data.shape
        features = features.view(shapes[0], shapes[1] * shapes[2] * shapes[3])  # features.shape torch.Size([32, 684])
        return features


class ConvsLayer2(BasicModule):
    def __init__(self, ):
        super(ConvsLayer2, self).__init__()

        self.kernels = configs.kernels
        hidden_channels = configs.cnn_chanel
        in_channel = 1
        features_L = configs.max_sequence_length
        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        W_size = seq_dim + dssp_dim + pssm_dim

        padding1 = (self.kernels[0] - 1) // 2
        padding2 = (self.kernels[1] - 1) // 2
        padding3 = (self.kernels[2] - 1) // 2
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding1, 0),
                                        kernel_size=(self.kernels[0], W_size)))  #
        self.conv1.add_module("BN", torch.nn.BatchNorm2d(228))
        self.conv1.add_module("Hardswish", nn.Hardswish())
        self.conv1.add_module("pooling1", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding2, 0),
                                        kernel_size=(self.kernels[1], W_size)))  #
        self.conv2.add_module("BN2", torch.nn.BatchNorm2d(228))
        self.conv2.add_module("Hardswish", nn.Hardswish())
        self.conv2.add_module("pooling2", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding3, 0),
                                        kernel_size=(self.kernels[2], W_size)))  #
        self.conv3.add_module("BN3", torch.nn.BatchNorm2d(228))
        self.conv3.add_module("Hardswish", nn.Hardswish())
        self.conv3.add_module("pooling3", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))


    def forward(self, x):

        features1 = self.conv1(x)  # features1.shape torch.Size([64, 228, 11, 1])
        features2 = self.conv2(x)
        features3 = self.conv3(x)  # features3.shape torch.Size([32, 228, 1, 1])

        features = t.cat((features1, features2, features3), 1)  # features.shape torch.Size([32, 684, 1, 1])

        shapes = features.data.shape
        features = features.view(shapes[0], shapes[1] * shapes[2] * shapes[3])  # features.shape torch.Size([32, 684])

        return features


class HNPPI(BasicModule):
    def __init__(self, class_nums, window_size, ratio=None):
        super(HNPPI, self).__init__()
        global configs
        configs.kernels = [13, 15, 17]
        self.dropout = configs.dropout = 0.2

        seq_dim = configs.seq_dim * configs.max_sequence_length

        self.seq_layers = nn.Sequential()
        self.seq_layers.add_module("seq_embedding_layer",
                                   nn.Linear(seq_dim, seq_dim))
        self.seq_layers.add_module("seq_embedding_ReLU",
                                   nn.ReLU())

        self.dssp_layers = nn.Sequential()
        self.dssp_layers.add_module("dssp_embedding_layer",
                                    nn.Linear(9 * 500, 9 * 500))
        self.dssp_layers.add_module("dssp_embedding_ReLU",
                                    nn.Hardswish())

        self.pssm_layers = nn.Sequential()
        self.pssm_layers.add_module("bilstm",
                                    nn.LSTM(20, 10, num_layers=3, dropout=0.1, bidirectional=True))
        self.bn_pssm = nn.BatchNorm1d(num_features=500)
        self.ra_pssm = ResAttn(channels=configs.batch_size)

        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        local_dim = (window_size * 2 + 1) * (pssm_dim + dssp_dim + seq_dim)
        if ratio:
            configs.cnn_chanel = (local_dim * int(ratio[0])) // (int(ratio[1]) * 3)
        input_dim = configs.cnn_chanel * 3 + local_dim

        self.multi_CNN = nn.Sequential()
        self.multi_CNN.add_module("layer_convs",
                                  ConvsLayer())

        self.multi_CNN2 = nn.Sequential()
        self.multi_CNN2.add_module("layer_convs2",
                                   ConvsLayer2())

        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("DNN_layer1",
                             nn.Linear(1027, 1024))  # 1711  # 2504  # 1027
        self.DNN1.add_module("ReLU1",
                             nn.Hardswish())
        # self.dropout_layer = nn.Dropout(self.dropout)
        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("DNN_layer2",
                             nn.Linear(1024, 256))
        self.DNN2.add_module("ReLU2",
                             nn.Hardswish())

        self.outLayer = nn.Sequential(
            # nn.Linear(256, class_nums),
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

        self.mixer7 = MLPMixer(
            image_size=7,
            channels=7,
            patch_size=1,  # 16
            dim=512,
            depth=2,  # 12
            num_classes=343  # 256
        )

        self.wmsa1 = WMSA(channels=configs.batch_size)
        self.wmsa2 = WMSA(channels=configs.batch_size)

        self.ra_gru = RAGRU(vocab_size=49, n_class=684, ifmha=True, rnn_hidden=49)

        self.ra = ResAttn(channels=configs.batch_size)


    def forward(self, seq, dssp, pssm, local_features):
        # dssp = self.logsig(dssp)
        # pssm = self.logsig(pssm)

        # ===================================================以下原始代码：nn=============================================
        shapes = seq.data.shape
        features = seq.view(shapes[0], shapes[1] * shapes[2] * shapes[3])  # torch.Size([32, 10000])
        features = self.seq_layers(features)
        features = features.view(shapes[0], shapes[1], shapes[2], shapes[3])

        # 尝试对dssp也使用linear
        dsspshapes = dssp.shape
        dssp_nn = dssp.view(dsspshapes[0], dsspshapes[1] * dsspshapes[2] * dsspshapes[3])  # torch.Size([32, 10000])
        dssp_nn = self.dssp_layers(dssp_nn)
        dssp_nn = dssp_nn.view(dsspshapes[0], dsspshapes[1], dsspshapes[2], dsspshapes[3])

        pssmshapes = pssm.shape
        pssm0 = pssm
        pssm_nn = pssm.reshape(pssmshapes[0], pssmshapes[1] * pssmshapes[2], pssmshapes[3])  # torch.Size([32, 10000])
        pssm_nn, _ = self.pssm_layers(pssm_nn)
        pssm_nn = self.bn_pssm(pssm_nn)
        pssm_nn = self.ra_pssm(pssm_nn)
        pssm_nn = pssm_nn.reshape(pssmshapes[0], pssmshapes[1], pssmshapes[2], pssmshapes[3])
        pssm_nn = pssm_nn * pssm0


        dssp_aff = self.wmsa1(dssp_nn.reshape(500, configs.batch_size, 1, 9))
        dssp_aff = dssp_aff.reshape(configs.batch_size, 1, 500, 9)

        features_aff = self.wmsa2(features.reshape(500, configs.batch_size, 1, 20))
        features_aff = features_aff.reshape(configs.batch_size, 1, 500, 20)

        features = t.cat((features_aff, dssp_aff, pssm_nn), 3)

        features_mcnn2 = self.multi_CNN2(features)
        features_mcnn1 = self.multi_CNN(features)
        features_mcnn = features_mcnn1 * features_mcnn2

        features_shape = features.shape

        features0, features_g_hidden0 = self.ra_gru(
            features.reshape(features_shape[0], features_shape[2], features_shape[3]))


        features = self.ra(features0.reshape(4, configs.batch_size, 171, 1))

        features = features.reshape(configs.batch_size, 684)

        features = features * features_mcnn

        local_features_m7 = self.mixer7(local_features.reshape(configs.batch_size, 7, 7, 7))
        local_features = local_features * local_features_m7


        features = t.cat((features, local_features), 1)

        features = self.DNN1(features)

        features = self.outLayer(features)

        return features

