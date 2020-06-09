# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 20:15
# @Author  : Equator
import csv
import numpy as np
from pandas import read_csv, set_option, DataFrame
import csv
from sklearn.model_selection import train_test_split


def scan(path):
    with open(path, 'r') as dataRows:
        while True:
            dataRow = dataRows.readline()
            if dataRow:
                yield dataRow
            else:
                break


def test_scan():
    dataRows = scan('../../data/agaricus-lepiota.data')
    print(dataRows)
    for dataRow in dataRows:
        # print(dataRow)
        print(dataRow[0], '#', dataRow[1:len(dataRow)])


def read_csv_file():
    with open('../../data/num_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = [row for row in reader]
        return np.asarray(rows, dtype=int)


def test_read_csv():
    data_rows = read_csv_file()
    print(data_rows)
    print(data_rows.shape)


def read_csv_pandas():
    names = get_names()
    data = read_csv('../../data/num_data.csv', names=names)
    return data


def get_names():
    return ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring',
            'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
            'spore-print-color', 'population', 'habitat']


def __char_to_int():
    names = get_names()
    df = read_csv('../../data/agaricus-lepiota.data', names=names)
    df.drop('stalk-root', axis=1, inplace=True)
    df.drop('veil-type', axis=1, inplace=True)
    df.drop('veil-color', axis=1, inplace=True)
    # print(df.shape)
    dataSet = []
    for d in df._values:
        data = []
        for cidx in range(len(d)):
            if cidx == 0:
                if d[cidx] == 'p':
                    data.append(0)
                else:
                    data.append(1)
            else:
                data.append(ord(d[cidx]) - ord('a'))
        dataSet.append(data)
    result = DataFrame(dataSet, columns=df.keys())
    f = open('../../data/data_preceded.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(result.keys())
    writer.writerows(result.values)


def get_total_data():
    data = read_csv('../../data/data_preceded.csv')
    return data


def data_split():
    data_set = get_total_data()
    arr = data_set.values
    x = arr[:, 1:arr.shape[1]]
    y = arr[:, 0]
    test_size = 0.3
    seed = 7
    # train_x,test_x,train_y,tets_y
    return train_test_split(x, y, test_size=test_size, random_state=seed)


if __name__ == '__main__':
    __char_to_int()
    get_total_data()
