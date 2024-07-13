import numpy as np
from mlpr_functions.visualizer import visualize_pairwise
from mlpr_functions.PCA import PCA
from mlpr_functions.LDA import LDA


def load_iris(path: str):
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
    return (matrix, label)


def main():
    D, L = load_iris('./src/iris.csv')
    classes = ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
    features = ['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)']


    ####################################
    # PCA - Principal Component Analysis

    y_PCA = PCA(D, 2)
    # x_PCA = P @ y_PCA # it isnt important

    visualize_pairwise(y_PCA, L, [0,1,2], classes)



    ####################################
    # LDA - Linear Discriminant Analysis

    y_LDA = LDA(y_PCA, L, [0,1], 2)
    visualize_pairwise(y_LDA, L, [0,1,2], classes)


if __name__ == '__main__':
    main()