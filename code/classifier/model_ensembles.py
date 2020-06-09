# -*- coding: utf-8 -*-
# @Time    : 2020/6/9 17:15
# @Author  : Equator
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from code.preprocessing import data_scan
from code.visual.algorithm_comparison import box_plot

num_folds = 10
seed = 7
scoring = 'accuracy'


def baseline(train_x, train_y):
    models = {}
    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier()
    models['NB'] = GaussianNB()
    models['SVM'] = SVC()
    models['LDA'] = QuadraticDiscriminantAnalysis()
    results = []
    for key in models:
        fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        result = cross_val_score(models[key], train_x, train_y, cv=fold, scoring=scoring)
        results.append(result)
        print("%s %f (%f)" % (key, result.mean(), result.std()))
        # print(result)
    box_plot(results, names=models.keys())


def score(train_x, train_y, test_x, test_y):
    models = {}
    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier()
    models['NB'] = GaussianNB()
    models['SVM'] = SVC()
    models['LDA'] = QuadraticDiscriminantAnalysis()
    for key in models:
        models[key].fit(train_x, train_y)
        s = models[key].score(test_x, test_y)
        print("%s ,%f" % (key, s))


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_scan.data_split()
    # score(train_x, train_y, test_x, test_y)
    baseline(train_x, train_y)
