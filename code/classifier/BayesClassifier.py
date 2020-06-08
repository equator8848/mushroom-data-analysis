# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 15:52
# @Author  : Equator
from code.classifier.BaseClassifier import BaseClassifier
from code.preprocessing.data_scan import read_csv_file
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from code.preprocessing.pca import PCA


class BayesClassifier(BaseClassifier):
    def __init__(self):
        self.classifier = GaussianNB()

    def train(self, train_data_x, train_data_y):
        self.classifier.fit(train_data_x, train_data_y)

    def classify(self, test_data_x):
        return self.classifier.predict(test_data_x)


def train_and_test():
    total_data = read_csv_file()
    total_size, attr_size = total_data.shape[0], total_data.shape[1]
    train_data = total_data[0:int(total_size * 2 / 3)]
    test_data = total_data[int(total_size * 2 / 3):total_size]
    # 训练集
    print(train_data.shape)
    train_data_x = train_data[:, 1:attr_size]
    train_data_y = train_data[:, 0]
    print(train_data_x.shape)
    print(train_data_y.shape)
    classifier = BayesClassifier()
    classifier.train(train_data_x, train_data_y)
    # 测试集
    print("验证测试集", test_data.shape)
    correct_size = 0
    for data in test_data:
        # print(data)
        test_data_x = data[1:attr_size]
        test_data_y = data[0]
        # print(data, test_data_x, test_data_y)
        result = classifier.classify(test_data_x.reshape(1, attr_size - 1))
        if result == test_data_y:
            correct_size = correct_size + 1
    print("正确率：%f%%" % (correct_size * 100 / test_data.shape[0]))


def train_and_test_with_pca():
    total_data = read_csv_file()
    total_size, attr_size = total_data.shape[0], total_data.shape[1]
    train_data = total_data[0:int(total_size * 2 / 3)]
    test_data = total_data[int(total_size * 2 / 3):total_size]
    # 训练集
    print(train_data.shape)
    train_data_x = train_data[:, 1:attr_size]
    train_data_y = train_data[:, 0]
    test_data_x = test_data[:, 1:attr_size]
    test_data_y = test_data[:, 0]
    c = 22
    train_data_x, att_idx = PCA(train_data_x, c, 1)
    test_data_x = PCA(test_data_x, c, 0)
    test_data_x = test_data_x[:, att_idx]
    print(train_data_x.shape)
    print(train_data_y.shape)
    classifier = BayesClassifier()
    classifier.train(train_data_x, train_data_y)
    # 测试集
    print("验证测试集", test_data.shape)
    correct_size = 0
    for i in range(len(test_data_x)):
        x = test_data_x[i]
        y = test_data_y[i]
        result = classifier.classify(x.reshape(1, c))
        if result == y:
            correct_size = correct_size + 1
    print("正确率：%f%%" % (correct_size * 100 / test_data.shape[0]))


if __name__ == '__main__':
    train_and_test_with_pca()
