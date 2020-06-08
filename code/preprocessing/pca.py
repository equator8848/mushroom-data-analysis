import numpy as np


def train_file_read():
    with open('train.csv', 'r') as f:
        reader = np.loadtxt(f, delimiter=',', skiprows=0)
        f.close()
        reader = np.array(reader)
        data = reader[:, 1:]
        label = reader[:, 0]
        return data, label


def test_file_read():
    with open('test.csv', 'r') as f:
        reader = np.loadtxt(f, delimiter=',', skiprows=0)
        f.close()
        reader = np.array(reader)
        data = reader[:, 1:]
        label = reader[:, 1]
        return data, label


def PCA(attribute, k, a):
    n_samples = attribute.shape[0]
    n_features = attribute.shape[1]

    mean = np.zeros((1, n_features))
    std_mean = np.zeros((n_samples, n_features))

    m = np.average(attribute, axis=0).reshape(1, n_features)
    for i in range(n_features):
        if m[0][i] is not None:
            mean[0][i] = m[0][i]
    for i in range(n_samples):
        for j in range(n_features):
            std_mean[i][j] = (attribute[i][j] - mean[0][j])
    if a == 0:
        return std_mean
    sigma = np.cov(std_mean, rowvar=0)
    value, vects = np.linalg.eig(sigma)
    eigVal_Ind = np.argsort(value)
    eigVal_Ind = eigVal_Ind[:-(k + 1):-1]
    eigvec_Ind = vects[:, eigVal_Ind]
    low_Attri = np.dot(std_mean, eigvec_Ind)

    # re_Attri = np.dot(low_Attri, eigvec_Ind.T)
    # red_Value = value[eigVal_Ind]

    # with open('train_data.csv', 'w') as ff:
    #     writer = csv.writer(ff, delimiter=',')
    #     writer.writerows(low_Attri)
    #     ff.close()
    return low_Attri, eigVal_Ind
