# v0.1 pytorch CNN 모델 검증

import numpy as np
import configparser
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

config = configparser.ConfigParser()
config.read('config.ini')


# https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py
class KISnet(nn.Module):
    def __init__(self):
        super(KISnet, self).__init__()

    # convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, 3, stride=1, padding=2),  # padding = 2 : same padding
            #   nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(2, 4, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(4, 8, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 16, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
    # fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(768 * 1 * 16, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU())
        self.fc4 = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU())
        self.output = nn.Linear(8, 2)

    def forward(self, x):
        out = F.dropout(self.layer1(x), p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = F.dropout(self.layer2(out), p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = F.dropout(self.layer3(out), p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = F.dropout(self.layer4(out), p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = x.view(-1, out.size(0))

        out = F.dropout(self.fc1(out), p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = F.dropout(self.fc2(out), p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = F.dropout(self.fc3(out), p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = F.dropout(self.fc4(out), p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = self.output(out)

        return F.log_softmax(out)


def initialize():
    # create model storage
    if not os.path.exists(config.get('CLASSIFIER', 'MODEL_STORAGE_P')):
        os.mkdir(config.get('CLASSIFIER', 'MODEL_STORAGE_P'))


class PEfileDataset(Dataset):
    # initialize data, download, etc.
    def __init__(self):
        self.file_lists, self.label_lists = list(), list()
        self.len = 0
        with open(config.get('PATH', 'TRAIN_MAL_FHS'), 'rb') as mal_file:
            self.mal_lists = pickle.load(mal_file)
            self.file_lists += self.mal_lists
            self.label_lists += [np.array([0, 1]) for _ in self.mal_lists]
            self.len += len(self.mal_lists)
        with open(config.get('PATH', 'TRAIN_BEN_FHS'), 'rb') as ben_file:
            self.ben_lists = pickle.load(ben_file)
            self.file_lists += self.ben_lists
            self.label_lists += [np.array([1, 0]) for _ in self.ben_lists]
            self.len += len(self.ben_lists)

    def __getitem__(self, index):
        return self.file_lists[index], self.label_lists[index]

    def __len__(self):
        return self.len


def run():
    initialize()

    # network architecture
    model = KISnet().cuda()

    cost_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('CLASSIFIER', 'LEARNING_RATE'))

    # train
    dataset = PEfileDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=int(config.get('CLASSIFIER', 'BATCH_SIZE')),
                              shuffle=True, num_workers=0)
    for e in range(int(config.get('CLASSIFIER', 'EPOCH'))):
        for batch_idx, train_mini_batch in enumerate(train_loader):
            data, label = train_mini_batch
            x = Variable(data).cuda()
            y_ = Variable(label).cuda()

            optimizer.zero_grad()
            output = model.forward(x)

            loss = cost_function(output, y_)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('loss : {}'.format(loss))
    pass


if __name__ == '__main__':
    run()
    pass