import numpy as np
import configparser
import csv
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

from spp_layer import get_spp_layer

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
            self.mal_file_path = config.get('PATH', 'EVAL_MAL_FHS')
            self.ben_file_path = config.get('PATH', 'EVAL_BEN_FHS')

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
            nn.Conv1d(1, 2, kernel_size=3, padding=1),  # padding = 1 : same padding
            #   nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU())
    # fully connected layer
        self.fc1 = nn.Linear(16 * int(config.get('CLASSIFIER', 'SPP_LEVEL_SUM')), 64, bias=True)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(64, 32, bias=True)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc3 = nn.Linear(32, 8, bias=True)
        nn.init.xavier_uniform(self.fc3.weight)
        self.output = nn.Linear(8, 2, bias=True)
        nn.init.xavier_uniform(self.output.weight)

    def forward(self, x, train_flag=True):
        drop_prob = float(config.get('CLASSIFIER', 'DROPOUT_PROB'))
        # convolutional layer
        out = self.layer1(x)
        if train_flag:
            out = F.dropout(out, p=drop_prob)
        out = self.layer2(out)
        if train_flag:
            out = F.dropout(out, p=drop_prob)
        out = self.layer3(out)
        if train_flag:
            out = F.dropout(out, p=drop_prob)
        out = self.layer4(out)
        out = get_spp_layer(out, list(out.size()), int(config.get('CLASSIFIER', 'SPP_LEVEL')))
        if train_flag:
            out = F.dropout(out, p=drop_prob)

        # fully connected layer
        out = self.fc1(out)
        out = F.relu(out)
        if train_flag:
            out = F.dropout(out, p=drop_prob)
        out = self.fc2(out)
        out = F.relu(out)
        if train_flag:
            out = F.dropout(out, p=drop_prob)
        out = self.fc3(out)
        out = F.relu(out)
        if train_flag:
            out = F.dropout(out, p=drop_prob)

        out = self.output(out)

        if train_flag:
            result = F.log_softmax(out)
        else:
            result = out

        return result


def train(step, log):
    print('# {}'.format(step))

    # load data
    start_time = time.time()
    print('data load start')
    train_dataset = PEfileDataset(train_flag=True)
    load_time = time.time() - start_time
    print('data load ended')
    print('data loading time: {} seconds'.format(load_time))

    with torch.cuda.device(int(config.get('CLASSIFIER', 'GPU_NUM'))):
        # network architecture
        model = KISnet().cuda()
        cost_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get('CLASSIFIER', 'LEARNING_RATE')))

        # model train
        batch_size = int(config.get('CLASSIFIER', 'BATCH_SIZE'))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print('--train start--')
        epoch = int(config.get('CLASSIFIER', 'EPOCH'))
        model_path = config.get('CLASSIFIER', 'MODEL_STORAGE_P')+str(step)
        start_time = time.time()
        for e in range(epoch):
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
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.3f' % (
                    e + 1, epoch, batch_idx + 1, len(train_loader.dataset) // batch_size, loss.data[0]))
                    torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), model_path)
        print('--train ended--')
        train_time = time.time() - start_time
        print('train time: {} seconds'.format(train_time))

    log += [step, load_time, train_time]
    pass


def evaluate(step, log):
    # load data
    start_time = time.time()
    print('data load start')
    test_dataset = PEfileDataset(train_flag=False)
    load_time = time.time() - start_time
    print('data load ended')
    print('data loading time: {} seconds'.format(load_time))

    with torch.cuda.device(int(config.get('CLASSIFIER', 'GPU_NUM'))):
        # network architecture
        model = KISnet().cuda()
        model.load_state_dict(torch.load(config.get('CLASSIFIER', 'MODEL_STORAGE_P')+str(step)))

        # model evaluate
        test_loader = DataLoader(dataset=test_dataset, batch_size=int(config.get('CLASSIFIER', 'BATCH_SIZE')),
                                 shuffle=False)

        print('--test start--')
        start_time = time.time()
        correct = 0
        for test_mini_batch in test_loader:
            data, label = test_mini_batch

            x = Variable(data,volatile=True).cuda()
            y_ = Variable(label).cuda()

            output = model.forward(x, train_flag=False)

            # get the index of the max log-probability
            pred_label = torch.max(output, 1)[1]
            correct += (pred_label == y_).sum().float()
        print('--test ended--')
        evaluation_time = time.time() - start_time
        total_accuracy = float(100. * correct / len(test_loader.dataset))
        print('test time: {} seconds'.format(evaluation_time))
        print('Accuracy: {}% ({}/{})\n'.format(total_accuracy, int(correct), len(test_loader.dataset)))

    log += [load_time, evaluation_time, total_accuracy]
    pass


def run():
    for step in range(1, 3):
        log = list()
        train(step, log)
        evaluate(step, log)
        with open(config.get('BASIC_INFO', 'TORCH_LOG_FILE_NAME'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(log)
    pass


if __name__ == '__main__':
    run()
    pass
