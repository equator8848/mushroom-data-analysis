# -*- coding: utf-8 -*-
# @Time    : 2020/6/3 17:32
# @Author  : Equator
from code.preprocessing.data_scan import read_csv_file, read_csv_pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_data():
    arr = read_csv_file()
    # data_x, data_y
    return arr[:, 1:len(arr)], arr[:, 0]


def scatter_visual():
    data_x, data_y = get_data()
    for i in range(data_x.shape[1]):
        plt.scatter(data_x[:, i], data_y)
        plt.xlabel("COL%d" % (i + 1))
        plt.ylabel("Classification")
        # plt.savefig("D:\\文件与文档\\学校课程\\数据挖掘\\数据挖掘实验\\mushroom-data-analysis\\report\\scatterplot%d.png" % (i + 1))
        plt.show()


# 散点图矩阵
def scatter_matrix_visual():
    data = read_csv_pandas()
    pd.plotting.scatter_matrix(data, figsize=(48, 48), diagonal='hist', marker='.')
    # plt.savefig("D:\\文件与文档\\学校课程\\数据挖掘\\数据挖掘实验\\mushroom-data-analysis\\report\\scatter_matrix_visual.png")
    plt.show()


if __name__ == '__main__':
    scatter_matrix_visual()
