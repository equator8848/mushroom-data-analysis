# -*- coding: utf-8 -*-
# @Time    : 2020/6/9 20:48
# @Author  : Equator
import numpy as np
import matplotlib.pyplot as plt
from code.preprocessing.data_scan import get_total_data


def figure(data):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    # 刻度
    ticks = np.arange(0, 21, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    names = list(data.columns)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


if __name__ == '__main__':
    data = get_total_data()
    figure(data)
