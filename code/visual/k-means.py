# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 11:08
# @Author  : Equator

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from code.preprocessing import data_scan


def kmeans():
    train_x, test_x, train_y, test_y = data_scan.data_split()
    pca = PCA(n_components=2)
    pca.fit(train_x)
    pca.transform(train_x)
    y_pred = KMeans(n_clusters=2, random_state=2).fit_predict(train_x)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=y_pred)
    plt.show()


if __name__ == '__main__':
    kmeans()
