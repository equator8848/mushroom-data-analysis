import numpy as np
from code.preprocessing.data_scan import data_split


def load_data(is_load_train_data):
    train_x, test_x, train_y, test_y = data_split(path='../../../data/data_preceded.csv')
    # 一定要标准化呀！！！
    threshold = 26
    if is_load_train_data:
        data = train_x
        labels = train_y
    else:
        data = test_x
        labels = test_y
    ls = []
    for label in labels:
        temp = np.zeros(2)
        temp[label] = 1
        ls.append(temp.T)
    labels = np.array(ls)
    return (data / threshold), labels


def load_total_data():
    train_x, test_x, train_y, test_y = data_split(path='../../data/data_preceded.csv')
    threshold = 26
    train = []
    for label in train_y:
        temp = np.zeros(2)
        temp[label] = 1
        train.append(temp.T)
    train_y = np.array(train)
    test = []
    for label in test_y:
        temp = np.zeros(2)
        temp[label] = 1
        test.append(temp.T)
    test_y = np.array(test)
    return (train_x / threshold), (test_x / threshold), train_y, test_y
