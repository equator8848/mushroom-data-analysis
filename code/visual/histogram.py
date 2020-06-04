# -*- coding: utf-8 -*-
# @Time    : 2020/6/3 17:32
# @Author  : Equator
from code.preprocessing.data_scan import read_csv_pandas
import numpy as np
import matplotlib.pyplot as plt


def histogram_visual():
    data = read_csv_pandas()
    data.hist()
    plt.show()


if __name__ == '__main__':
    histogram_visual()
