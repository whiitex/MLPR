from matplotlib import pyplot as plt
import numpy as np

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

    # centered matrix
    mean = D.mean(1).reshape(D.shape[0], 1)
    DC = D - mean

    mu = []
    for i in range(3):
        mu.append(D[:, L==i].mean(1))
    mu = np.array(mu).T
    print(mu)


    features = ['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)']

    fig, axs = plt.subplots(4,4, figsize=(20,16))
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            if i == j:
                axs[i][j].hist(D[i, L==0], density=1, label='Setosa', alpha=0.5)
                axs[i][j].hist(D[i, L==1], density=1, label='Versicolor', alpha=0.5)
                axs[i][j].hist(D[i, L==2], density=1, label='Virginica', alpha=0.5)
                # axs[i][j].plot([mu[i,0], mu[i,0]], [0,2])
                # axs[i][j].plot([mu[i,1], mu[i,1]], [0,2])
                # axs[i][j].plot([mu[i,2], mu[i,2]], [0,2])
                axs[i][j].set_title(features[i], fontsize=10)
            else:
                axs[i][j].scatter(D[i, L==0], D[j, L==0], label='Setosa', alpha=0.65, marker='.')
                axs[i][j].scatter(D[i, L==1], D[j, L==1], label='Versicolor', alpha=0.65, marker='.')
                axs[i][j].scatter(D[i, L==2], D[j, L==2], label='Virginica', alpha=0.65, marker='.')
                axs[i][j].set_title(features[i] + ' x ' + features[j], fontsize=10)
            axs[i][j].legend()

    fig.tight_layout(pad=4)
    plt.show()


    var = DC.var(1)
    std = DC.std(1)

    print(var)
    print(std)


if __name__ == '__main__':
    main()