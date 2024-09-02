import numpy as np
from mlpr_functions.data_management import *
from mlpr_functions.SVM import *
from mlpr_functions.BayesRisk import *


def load_iris_binary(path: str):
    file = open(path, 'r')
    matrix = []
    label = []
    for line in file:
        line = line.split(sep=',')
        matrix.append(line[0:4])
        name : int
        if line[4] == 'Iris-setosa\n': name = 0
        elif line[4] == 'Iris-versicolor\n': name = 1
        else: name = 2
        label.append(name)
    
    matrix = np.array(matrix, dtype='float32').T
    label = np.array(label, dtype=int)
    file.close()

    matrix = matrix[:, label != 0]
    label = label[label != 0]
    label[label == 2] = 0
    return (matrix, label)


def main():

#-------------------------------------------------------------------#

    # Iris dataset
    classes = [0,1,2]
    D, L = load_iris_binary('iris.csv')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)



#-------------------------------------------------------------------#

    # SVM
    C, K = 1, 10.0
    w, b, res = train_dual_SVM_linear(DTR, LTR, C, K)
    print(f'PL = {res[0]:e}, DL = {res[1]:e}, DG = {res[2]:e}')
    SVAL = (vrow(w) @ DTE + b).ravel()
    SVAL[SVAL > 0] = 1
    SVAL[SVAL <= 0] = 0
    accuracy = np.mean(SVAL == LTE)
    print(f'Error rate: {100 -accuracy * 100:.1f}%')



#-------------------------------------------------------------------#

    # SVM kernel function
    eps, C = 0.0,  1.0
    degree, c = 2, 1.0
    fScore, dualLoss = train_dual_SVM_kernel(DTR, LTR, C, polynomialKernel(degree, c), eps)
    print(f'Dual loss: {dualLoss:e} - ', end='')
    SVAL, PVAL = fScore(DTE), fScore(DTE)
    PVAL[PVAL > 0] = 1
    PVAL[PVAL <= 0] = 0
    accuracy = np.mean(PVAL == LTE)
    print(f'Error rate: {100 -accuracy * 100:.1f}%')
    print(f'minDCF: {compute_minDCF_binary(SVAL, LTE, 0.5, 1, 1)[0]:.4f}')
    print(f'actDCF: {compute_bayes_risk_binary(PVAL, LTE, 0.5, 1, 1)[1]:.4f}')



#-------------------------------------------------------------------#


if __name__ == '__main__':
    main()