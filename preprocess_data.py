import pickle
import numpy as np
from pathlib import Path
import torch as t
from torch import device


def normalization(seqData, max, min):
    return (seqData - min) / (max - min)


def standardization(seqData, mean, std):
    return (seqData - mean) / std


def reconstruct(seqData, mean, std):
    return seqData * std + mean


class PickleDataLoad(object):
    def __init__(self, data_type, filename, window_size, augment_test_data=True):
        # window_size = input_window_size + pred_window_size
        self.window_size = window_size
        self.augment_test_data = augment_test_data
        self.trainData, self.trainLabel = self.preprocessing(Path('dataset', data_type, 'labeled', 'train', filename),
                                                             train=True)
        self.testData, self.testLabel = self.preprocessing(Path('dataset', data_type, 'labeled', 'test', filename),
                                                           train=False)

    def augmentation(self, data, label, nosie_rate=0.03, N=10000):
        '''

        :param data:
        :param label:
        :param N: data size
        :return: data[N,window_size,channel] label[N]
        '''

        tar_pos = t.randint(self.window_size, self.length, [N], dtype=t.int)
        augmentedData = t.zeros([N, self.window_size, data.size(1)])
        augmentedLabel = t.zeros([N, 1])
        for i in range(N):
            X = data[tar_pos[i] - self.window_size:tar_pos[i]]
            noise_seq = t.randn(X.size())
            noise_seq *= self.std.expand_as(X)
            noise_seq *= nosie_rate
            augmentedData[i] = X + noise_seq
            augmentedLabel[i] = label[tar_pos[i]]

        return augmentedData, augmentedLabel

    def serializing(self, data, label):
        '''

        :param data:
        :param label:
        :return: data[N,window_size,channel] label[N]
        '''
        N = self.length - self.window_size + 1
        seqData = t.zeros([N, self.window_size, data.size(1)])
        seqLabel = t.zeros([N, 1])
        for i in range(N):
            seqData[i] = data[i:i + self.window_size]
            seqLabel[i] = label[i + self.window_size - 1]

        return seqData, seqLabel

    def preprocessing(self, path, train=True):
        '''Read,Standardize'''

        with open(str(path), 'rb') as f:
            data = t.FloatTensor(pickle.load(f))
            label = data[:, -1]
            data = data[:, :-1]

        self.length = len(data)

        if train:
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            assert self.length > self.window_size, "window size must < seq data length"
            print('train data length:', self.length)
            data, label = self.augmentation(data, label)

        else:
            print('test data length:', self.length)
            if self.augment_test_data:
                data, label = self.augmentation(data, label, N=5000)
            else:
                data, label = self.serializing(data, label)

        data = standardization(data, self.mean, self.std)

        return data.transpose(1, 2), label

    def batchify(self, args, data, bsz, isLabel=False):
        nbatch = data.size(0) // bsz
        trimmed_data = data.narrow(0, 0, nbatch * bsz)
        if not isLabel:
            batched_data = trimmed_data.contiguous().view(bsz, -1, trimmed_data.size(-2),
                                                          trimmed_data.size(-1)).transpose(0, 1)
        else:
            batched_data = trimmed_data.contiguous().view(-1, trimmed_data.size(-1))
        batched_data = batched_data.to(device(args.device))
        return batched_data
