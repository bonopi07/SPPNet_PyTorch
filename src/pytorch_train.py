# v0.1 pytorch CNN 모델 검증

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py
class KISnet(nn.Module):
    def __init__(self):
        super(KISnet.self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, stride=1, padding=2)  # padding = 2 : same padding
        self.conv2 = nn.Conv1d(2, 4, 3, stride=1, padding=2)
        self.conv3 = nn.Conv1d(4, 8, 3, stride=1, padding=2)
        self.conv4 = nn.Conv1d(8, 16, 3, stride=1, padding=2)
        # self.act1 = nn.ReLU() # self.pool1 = nn.MaxPool2d(2,2)

        self.dense1 = nn.Linear(816 * 1 * 16, 4096) # need to modify
        self.dense2 = nn.Linear(4096, 512)
        self.dense2 = nn.Linear(512, 64)
        self.dense2 = nn.Linear(64, 8)
        self.dense2 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 64*3*3)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, training=self.training)
        x = self.dense2(x)

        return F.log_softmax(x)


def load_data():
    pass


def init():
    if not os.path.exists(MODEL_SNAPSHOT):
        os.makedirs(MODEL_SNAPSHOT)
    pass


def run():
    init()
    mal_lists, ben_lists = load_data()

    model = KISnet().cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train
    for i in range(ITER):
        for j, [image, label] in enumerate():


    pass


if __name__ == '__main__':
    run()
    pass