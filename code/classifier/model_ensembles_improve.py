# -*- coding: utf-8 -*-
# @Time    : 2020/6/9 17:15
# @Author  : Equator
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from code.preprocessing import data_scan
from code.visual.algorithm_comparison import box_plot

num_folds = 10
seed = 7
scoring = 'accuracy'


def scaler(train_x, train_y):
    piplelines = {}
    piplelines['ScalerKNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])
    piplelines['ScalerCART'] = Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])
    piplelines['ScalerNB'] = Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])
    piplelines['ScalerSVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])
    piplelines['ScalerLDA'] = Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])
    results = []
    for key in piplelines:
        fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        result = cross_val_score(piplelines[key], train_x, train_y, cv=fold, scoring=scoring)
        results.append(result)
        print("%s %f (%f)" % (key, result.mean(), result.std()))
        # print(result)
    box_plot(results, names=piplelines.keys())

if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_scan.data_split()
    # score(train_x, train_y, test_x, test_y)
    scaler(train_x, train_y)
