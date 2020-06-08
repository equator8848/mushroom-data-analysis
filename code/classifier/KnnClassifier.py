from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from code.classifier.BaseClassifier import BaseClassifier
from code.preprocessing.data_scan import read_csv_file


class KnnClassifier(BaseClassifier):
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=8, algorithm='auto')

    def train(self, train_data_x, train_data_y):
        self.classifier.fit(train_data_x, train_data_y)

    def classify(self, test_data_x):
        return self.classifier.predict(test_data_x)


