import numpy as np
import scipy
from mlpr_functions.data_management import *
from mlpr_functions.LogisticRegression import *


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

    # Minimun optimization function with `scipy.optimize.fmin_l_bfgs_b`

    # def func(x: np.array):
    #     y = (x[0] + 3) ** 2 + np.sin(x[0]) + (x[1] + 1) ** 2
    #     d = np.zeros(2)
    #     d[0] = 2*(x[0] + 3) + np.cos(x[0])
    #     d[1] = 2*(x[1] + 1)
    #     return y, d

    # x, f, d = scipy.optimize.fmin_l_bfgs_b(func, x0=[0,-0])
    # print(x)
    # print(f'Gradient value: {d['grad']}')
    # print(f'Function calls: {d['funcalls']}')
    # print(f'Iterations: {d['nit']}')


#-------------------------------------------------------------------#

    # Iris dataset
    classes = [0,1,2]
    D, L = load_iris_binary('src/iris.csv')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)


#-------------------------------------------------------------------#

    # Binary Logistic Regression

    # best lamda to use: 0.0073
    w, b = trainLogRegBinary(DTR, LTR, 0.001)
    VAL = w.T @ DTE + b
    VAL[VAL > 0] = 1
    VAL[VAL < 0] = 0
    print(f'Accuracy: {np.mean(VAL == LTE) * 100:.2f}%')
    print(f'Error rate: {np.mean(VAL != LTE) * 100:.2f}%')


#-------------------------------------------------------------------#

    # Weighted Binary Logistic Regression

    # best lamda to use: 0.0073
    # best prior to use: [0.15, 0.45]
    wp, bp = trainWeightedLogRegBinary(DTR, LTR, 0.01, 0.30)
    VALp = wp.T @ DTE + bp
    VALp[VALp > 0] = 1
    VALp[VALp < 0] = 0
    print(f'Accuracy: {np.mean(VALp == LTE) * 100:.2f}%')
    print(f'Error rate: {np.mean(VALp != LTE) * 100:.2f}%')

#-------------------------------------------------------------------#




if __name__ == '__main__':
    main()