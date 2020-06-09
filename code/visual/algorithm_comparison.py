# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 15:47
# @Author  : Equator
import matplotlib.pyplot as plt


def box_plot(results, names):
    fig = plt.figure()
    fig.suptitle('AlgorithmComparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()