import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN_MAL_DIR_PATH = os.path.normpath(os.path.abspath('../data/train/mal'))
TRAIN_BEN_DIR_PATH = os.path.normpath(os.path.abspath('../data/train/ben'))
MODEL_SNAPSHOT = os.path.normpath(os.path.abspath('../data/train/MODEL_SNAPSHOT'))

BATCH_SIZE = int(128)
LEARNING_RATE = 1e-4
ITER = int(1000)

# https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py
class KISnet(nn.Module):
    def __init__(self):
        super(KISnet.self).__init__()
        self.conv1 =  nn.Conv2d(1, 16, 5, stride=1, padding=2) # padding = 2 : same padding
        # self.act1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        # self.act2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        # self.act3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(2, 2)

        self.dense1 = nn.Linear(64*3*3, 100) # need to modify
        # self.act4 = nn.ReLU()
        self.dense2 = nn.Linear(100, 10)

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