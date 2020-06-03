# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 20:15
# @Author  : Equator
import csv
import numpy as np


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


def read_csv():
    with open('../../data/num_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = [row for row in reader]
        return np.asarray(rows, dtype=int)


def test_read_csv():
    data_rows = read_csv()
    print(data_rows)
    print(data_rows.shape)


if __name__ == '__main__':
    test_read_csv()
