from sklearn.tree import DecisionTreeClassifier

class Decision_treeClassifier(BaseClassifier):
    def __init__(self):
        self.classifier = DecisionTreeClassifier()

    def train(self,train_data_x, train_data_y):
        self.classifier.fit(train_data_x, train_data_y)

    def classify(self,test_data_x):
        return self.classifier.predict(test_data_x)