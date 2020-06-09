from sklearn.tree import DecisionTreeClassifier

from code.classifier.BaseClassifier import BaseClassifier
from code.preprocessing import data_scan


class MyDecisionTreeClassifier(BaseClassifier):
    def __init__(self):
        self.classifier = DecisionTreeClassifier()

    def train(self, train_data_x, train_data_y):
        self.classifier.fit(train_data_x, train_data_y)

    def classify(self, test_data_x):
        return self.classifier.predict(test_data_x)


def train_and_test():
    train_x, test_x, train_y, test_y = data_scan.data_split()
    classifier = MyDecisionTreeClassifier()
    classifier.train(train_x, train_y)
    # 测试集
    print("验证测试集", train_x.shape)
    correct_size = 0
    for i in range(len(test_y)):
        # print(data)
        test_data_x = test_x[i]
        test_data_y = test_y[i]
        # print(data, test_data_x, test_data_y)
        result = classifier.classify(test_data_x.reshape(1, 20))
        print(result[0], test_data_y)
        if result[0] == test_data_y:
            correct_size = correct_size + 1
    print("正确率：%f%%" % (correct_size * 100 / test_x.shape[0]))


if __name__ == '__main__':
    train_and_test()
