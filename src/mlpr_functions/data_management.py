import numpy as np


def load_data(path: str):
    file = open(path, 'r')
    matrix = []
    label = []
    for line in file:
        line = line.split(',')
        for i in range(6):
            line[i] = float(line[i].strip())
        line[6] = int(line[6].strip())
        matrix.append(line[0:6])
        label.append(line[len(line)-1])
    matrix = np.array(matrix).T
    label = np.array(label)
    return (matrix, label)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def vcol(x: np.array):
    return x.reshape((x.size, 1))

def vrow(x: np.array):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C