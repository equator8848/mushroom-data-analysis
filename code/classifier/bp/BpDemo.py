from code.classifier.bp import DataSet
from code.classifier.bp.NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    train_times = 32
    learn_step = 0.1
    layers = [20, 40, 2]
    network = NeuralNetwork(2, learn_step, layers)
    x_train, y_train = DataSet.load_data(True)
    print("训练环节...")
    for i in range(train_times):
        for j in range(len(x_train)):
            network.update(x_train[j], y_train[j])
    correct_num = 0
    print("测试环节...")
    x_test, y_test = DataSet.load_data(False)
    for i in range(len(y_test)):
        if network.test(x_test[i], y_test[i]):
            correct_num += 1
    print("正确率 %f%%" % (correct_num * 100 / len(y_test)))
