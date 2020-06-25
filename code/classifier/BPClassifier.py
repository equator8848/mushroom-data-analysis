# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 21:56
# @Author  : Equator
from code.classifier.BaseClassifier import BaseClassifier
from code.classifier.bp import DataSet
from code.classifier.bp.NeuralNetwork import NeuralNetwork
from code.preprocessing import data_scan
from code.visual.score import visual
import numpy as np


class BPClassifier(BaseClassifier):
    def __init__(self):
        self.train_times = 8
        learn_step = 0.1
        layers = [20, 40, 2]
        self.network = NeuralNetwork(2, learn_step, layers)

    def train(self, train_data_x, train_data_y):
        for i in range(self.train_times):
            for j in range(len(train_data_x)):
                self.network.update(train_data_x[j], train_data_y[j])

    def classify(self, test_data_x):
        return self.network.classify(test_data_x)


def get_score():
    correct_size = 0
    for i in range(len(test_y)):
        test_data_x = test_x[i]
        test_data_y = test_y[i]
        temp = np.zeros(2)
        temp[test_data_y] = 1
        test_data_y = np.array(temp)
        result = classifier.classify(test_data_x.reshape(1, 20))
        if result == list(test_data_y).index(1):
            correct_size = correct_size + 1
    score = correct_size * 100 / test_x.shape[0]
    return score


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_scan.data_split()
    classifier = BPClassifier()
    classifier.train(train_x, train_y)
    print('测试准确率%f' % get_score())
