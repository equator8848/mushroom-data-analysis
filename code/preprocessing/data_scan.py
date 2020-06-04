# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 20:15
# @Author  : Equator
import csv
import numpy as np
from pandas import read_csv


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
    data = read_csv('../../data/num_data.csv', names=names, engine='python')
    return data


def get_names():
    return ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring',
            'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
            'spore-print-color', 'population', 'habitat']


def test_read_csv_pandas():
    data = read_csv_pandas()
    print(data)


if __name__ == '__main__':
    test_read_csv_pandas()
