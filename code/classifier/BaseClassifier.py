# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 16:59
# @Author  : Equator

# 分类器基类，每个分类器均需要继承该基类，便于后续的组合
class BaseClassifier:
    # 分类方法 输入一个 n*1 的向量，输出test_data_y即分类标签
    def classify(self, test_data_x):
        pass
