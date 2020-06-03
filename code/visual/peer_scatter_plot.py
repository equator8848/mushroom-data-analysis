# -*- coding: utf-8 -*-
# @Time    : 2020/6/3 17:32
# @Author  : Equator
from code.preprocessing.DataScanner import read_csv
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    arr = read_csv()
    # data_x, data_y
    return arr[:, 1:len(arr)], arr[:, 0]


def visual():
    data_x, data_y = get_data()
    for i in range(data_x.shape[1]):
        plt.scatter(data_x[:, i], data_y)
        plt.xlabel("COL%d" % (i + 1))
        plt.ylabel("Classification")
        # plt.savefig("D:\\文件与文档\\学校课程\\数据挖掘\\数据挖掘实验\\mushroom-data-analysis\\report\\scatterplot%d.png" % (i + 1))
        plt.show()


if __name__ == '__main__':
    visual()
