# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 8:52
# @Author  : Equator
from code.classifier.BPClassifier import BPClassifier
from code.classifier.MyDecisionTreeClassifier import MyDecisionTreeClassifier
from code.classifier.KnnClassifier import KnnClassifier

from code.preprocessing.data_scan import data_split

if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_split()
    classifiers = {}
    classifiers['BPNetWork'] = BPClassifier()
    classifiers['CART'] = MyDecisionTreeClassifier()
    classifiers['KNN'] = KnnClassifier()
    # 训练与构建模型
    for key in classifiers:
        classifiers[key].train(train_x, train_y)
    print('模型训练完毕...')
    correct_num = 0
    for i in range(len(test_y)):
        test_data_x = test_x[i]
        test_data_y = test_y[i]
        output = []
        for key in classifiers:
            val = classifiers[key].classify(test_data_x.reshape(1, 20))
            output.append(val)
        print('模型输出', output, '测试数据标签', test_data_y)
        # 求出出现次数最多的数字
        result = max(set(output), key=output.count)
        if result == test_data_y:
            correct_num += 1
    print("正确率：%f%%" % (correct_num * 100 / test_x.shape[0]))
