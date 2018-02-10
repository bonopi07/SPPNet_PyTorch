import numpy as np
import configparser
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

config = configparser.ConfigParser()
config.read('config.ini')


class PEfileDataset(Dataset):
    # initialize data, download, etc.
    def __init__(self, train_flag=True):
        # variable initialization
        self.file_lists, self.label_lists = list(), list()
        self.len = 0

        # decision about train mode or test mode (dataset)
        if train_flag:
            self.mal_file_path = config.get('PATH', 'TRAIN_MAL_FHS')
            self.ben_file_path = config.get('PATH', 'TRAIN_BEN_FHS')
        else:
            self.mal_file_path = config.get('PATH', 'TEST_MAL_FHS')
            self.ben_file_path = config.get('PATH', 'TEST_BEN_FHS')

        # load data,labels from pickle file
        with open(self.mal_file_path, 'rb') as mal_file:
            self.mal_lists = pickle.load(mal_file)
            for mal in self.mal_lists:
                self.file_lists.append(torch.Tensor(np.array(mal)).view(1, -1))
            self.label_lists += [1 for _ in self.mal_lists]
            self.len += len(self.mal_lists)
        with open(self.ben_file_path, 'rb') as ben_file:
            self.ben_lists = pickle.load(ben_file)
            for ben in self.ben_lists:
                self.file_lists.append(torch.Tensor(np.array(ben)).view(1, -1))
            self.label_lists += [0 for _ in self.ben_lists]
            self.len += len(self.ben_lists)

    def __getitem__(self, index):
        return self.file_lists[index], self.label_lists[index]

    def __len__(self):
        return self.len


# https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py
class KISnet(nn.Module):
    def __init__(self):
        super(KISnet, self).__init__()

    # convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=3, padding=2),  # padding = 2 : same padding
            #   nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
    # fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(769 * 1 * 16, 4096),
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

    def forward(self, x, train_flag=True):
        out = self.layer1(x)
        if train_flag:
            out = F.dropout(out, p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = self.layer2(out)
        if train_flag:
            out = F.dropout(out, p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = self.layer3(out)
        if train_flag:
            out = F.dropout(out, p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = self.layer4(out)
        if train_flag:
            out = F.dropout(out, p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))

        out = out.view(-1, 1 * 16 * 769)  # why 769? I think 768

        out = self.fc1(out)
        if train_flag:
            out = F.dropout(out, p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = self.fc2(out)
        if train_flag:
            out = F.dropout(out, p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = self.fc3(out)
        if train_flag:
            out = F.dropout(out, p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = self.fc4(out)
        if train_flag:
            out = F.dropout(out, p=float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
        out = self.output(out)

        if train_flag:
            result = F.log_softmax(out)
        else:
            result = out

        return result


def put_log(sequence, log):
    print(sequence)
    log.append(sequence + '\n')


def train(step):
    log = list()
    put_log('# {}'.format(step), log=log)

    # load data
    s_time = time.time()
    print('data load start')
    train_dataset = PEfileDataset(train_flag=True)
    print('data load ended')
    put_log('data loading time: {} seconds'.format(time.time() - s_time), log=log)

    # network architecture
    model = KISnet().cuda()
    cost_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get('CLASSIFIER', 'LEARNING_RATE')))

    # model train
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(config.get('CLASSIFIER', 'BATCH_SIZE')),
                              shuffle=True, num_workers=0)
    put_log('--train start--', log=log)
    s_time = time.time()
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
                put_log('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (e + 1, int(config.get('CLASSIFIER', 'EPOCH')), batch_idx + 1,
                         len(train_loader.dataset) // int(config.get('CLASSIFIER', 'BATCH_SIZE')), loss.data[0]), log=log)
                torch.save(model.state_dict(), config.get('CLASSIFIER', 'MODEL_STORAGE_P')+str(step))
    torch.save(model.state_dict(), config.get('CLASSIFIER', 'MODEL_STORAGE_P')+str(step))
    put_log('--train ended--', log=log)
    put_log('train time: {} seconds'.format(time.time() - s_time), log=log)

    with open(config.get('BASIC_INFO', 'LOG_FILE_NAME'), 'a') as f:
        for line in log:
            f.write(line)
    pass


def evaluate(step):
    log = list()

    # load data
    s_time = time.time()
    print('data load start')
    test_dataset = PEfileDataset(train_flag=False)
    print('data load ended')
    put_log('data loading time: {} seconds'.format(time.time() - s_time), log=log)

    # network architecture
    model = KISnet().cuda()
    model.load_state_dict(torch.load(config.get('CLASSIFIER', 'MODEL_STORAGE_P')+str(step)))

    # model evaluate
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(config.get('CLASSIFIER', 'BATCH_SIZE')),
                             shuffle=False)

    put_log('--test start--', log=log)
    s_time = time.time()
    correct = 0
    for test_mini_batch in test_loader:
        data, label = test_mini_batch

        x = Variable(data,volatile=True).cuda()
        y_ = Variable(label).cuda()

        output = model.forward(x, train_flag=False)

        # get the index of the max log-probability
        pred_label = torch.max(output, 1)[1]
        correct += (pred_label == y_).sum().float()
    put_log('--test ended--', log=log)
    put_log('test time: {} seconds'.format(time.time() - s_time), log=log)

    put_log('Accuracy: {}% ({}/{})\n'.format(float(100. * correct / len(test_loader.dataset)), int(correct),
                                           len(test_loader.dataset)), log=log)

    with open(config.get('BASIC_INFO', 'LOG_FILE_NAME'), 'a') as f:
        for line in log:
            f.write(line)
    pass


def run():
    for step in range(1, 20):
        train(step)
        evaluate(step)
    pass


if __name__ == '__main__':
    run()
    pass