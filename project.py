import numpy as np
from matplotlib import pyplot as plt
from src.mlpr_functions.visualizer import visualize_pairwise
from src.mlpr_functions.PCA import PCA
from src.mlpr_functions.LDA import LDA
from src.mlpr_functions.logpdf_GAU_ND import logpdf_GAU_ND

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


def main():
    D, L = load_data('./trainData.txt')
    classes = ['Counterfeit', 'Genuine']
    
    mu = D.mean(axis=1).reshape(D.shape[0], 1)
    DM = D - mu

    # Pair-wise scatter plots dependencies
    # visualize_pairwise(DM, L, np.array([0,1]))


    ############################################################
    # PCA - Principal Component Analysis

    m_PCA = 4 # 4/5 should be good
    y_PCA = PCA(D, m_PCA)
    # visualize_pairwise(y_PCA, L, [0,1], classes, a=0.05, bins=40)



    ############################################################
    # LDA - Linear Discriminant Analysis

    m_LDA = 3
    y_LDA = LDA(y_PCA, L, [0,1], m_LDA)
    # visualize_pairwise(y_LDA, L, [0,1], classes, a=0.05)



    ############################################################
    # Gaussian Multivariate density fitting

    fig, axs = plt.subplots(4, 3, figsize=(20,16))
    for c in range(2):
        Dc = np.sort(D[:, L == c], axis=1)
        mu = Dc.mean(axis=1).reshape(Dc.shape[0], 1)
        for i in range(D.shape[0]):
            row = Dc[i,:].reshape(1, Dc.shape[1])
            Sigma = (row - mu[i]) @ (row - mu[i]).T / row.shape[1]
            Sigma = np.ones((1,1)) * Sigma
            axs[c*2 + i//3][i%3].hist(row.ravel(), label=classes[c], density=1, bins=50, alpha=.8)
            axs[c*2 + i//3][i%3].plot(row.ravel(), np.exp(logpdf_GAU_ND(row, mu[i], Sigma)), linewidth=1.75)
            axs[c*2 + i//3][i%3].set_title(f"Feature {i+1}", fontsize=10)
            axs[c*2 + i//3][i%3].legend(fontsize=8)

    fig.tight_layout(pad=5)
    plt.show()




if __name__ == '__main__':
    main()