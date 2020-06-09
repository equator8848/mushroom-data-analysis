# -*- coding: utf-8 -*-
# @Time    : 2020/6/3 17:32
# @Author  : Equator
from code.preprocessing.data_scan import get_total_data
import numpy as np
import matplotlib.pyplot as plt


def histogram_visual():
    data = get_total_data()
    data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(16, 10))
    # plt.savefig("D:\\文件与文档\\学校课程\\数据挖掘\\数据挖掘实验\\mushroom-data-analysis\\report\\histogram_visual_processed.png")
    plt.show()


if __name__ == '__main__':
    histogram_visual()
