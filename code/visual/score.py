# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 22:46
# @Author  : Equator

import matplotlib.pyplot as plt


def visual(x, y):
    plt.plot(x, y, 'go-')
    plt.title('Accuracy')
    plt.xlabel('index')
    plt.ylabel('accuracy')
    plt.show()
