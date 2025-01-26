import numpy as np
import matplotlib.pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *
from src.mlpr_functions.PCA import *
from src.mlpr_functions.LDA import *


def main(m_PCA, m_LDA, center: bool):
    D, L = load_data('data_et_checks/trainData.txt')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    classes = ['Counterfeit', 'Genuine']


    if center:
        DTR = DTR - compute_mu_C(DTR)[0] # cetering data
        DTE = DTE - compute_mu_C(DTR)[0] # cetering data
    
    #########################################################################
    # PCA - Principal Component Analysis


    fig, axs = plt.subplots(1, 3, figsize=(15,4.5))
    for j in range(3):
        axs[j].scatter(DTR[2*j, LTR==0], DTR[2*j+1, LTR==0], label=classes[0], marker='.', alpha=0.24)
        axs[j].scatter(DTR[2*j, LTR==1], DTR[2*j+1, LTR==1], label=classes[1], marker='.', alpha=0.24)

        axs[j].set_title(f"Feature {2*j+1} x Feature {2*(j+1)}", fontsize=12)
        axs[j].legend(fontsize=10)

    fig.tight_layout(pad=3)
    plt.show()
    # plt.savefig('latex/images/feat_12_34_56.pdf', format='pdf')


    # m_PCA = 6 # 4/5 should be good
    P_PCA, DTR_PCA = PCA(DTR, m_PCA)
    DTE_PCA = P_PCA.T @ DTE



    visualize_pairwise(DTR_PCA, LTR, [0,1], classes, a=0.1, bins=40)


    # fig, axs = plt.subplots(2, 3, figsize=(16,9))
    # matrix = P_PCA.T @ DTR
    # for i in range(2):
    #     for j in range(3):
    #         for cls in range(len(classes)):
    #             axs[i][j].hist(matrix[i*3 + j, LTR==cls], density=1, bins=35, label=classes[cls], alpha=0.65)
    #         axs[i][j].set_title(f"Feature {i}", fontsize=10)
    #         axs[i][j].legend(fontsize=8)

    # fig.tight_layout(pad=3)
    # plt.show()
    # # plt.savefig('latex/images/PCA_6.pdf', format='pdf')


    #########################################################################
    # LDA - Linear Discriminant Analysis 
    
    # m_LDA = 3
    W, DTR_LDA = LDA(DTR, LTR, [0,1], min(m_PCA, m_LDA), True)
    DTE_LDA = W.T @ DTE

    # PCA vs LDA
    fig, axs = plt.subplots(1, 2, figsize=(10.6,5))
    for cls in range(len(classes)):
        axs[0].hist(DTR_PCA[0, LTR==cls], density=1, bins=45, label=classes[cls], alpha=0.65)
    axs[0].set_title(f"PCA", fontsize=12)
    axs[0].legend(fontsize=8)
 
    for cls in range(len(classes)):
        axs[1].hist(DTR_LDA[0, LTR==cls], density=0, bins=45, label=classes[cls], alpha=0.65)
    axs[1].set_title(f"LDA", fontsize=12)
    axs[1].legend(fontsize=8)

    fig.tight_layout(pad=3)
    plt.show()
    # plt.savefig('latex/images/PCAvsLDA.pdf', format='pdf')

    visualize_pairwise(vrow(DTR_LDA[0, :]), LTR, [0,1], classes, a=0.1) 

    mu0 = W[:, 0].T @ DTR[:, LTR == 0].mean(axis=1)
    mu1 = W[:, 0].T @ DTR[:, LTR == 1].mean(axis=1)

    if mu1 < mu0:
        W = -W
        mu0 = W[:, 0].T @ DTR[:, LTR == 0].mean(axis=1)
        mu1 = W[:, 0].T @ DTR[:, LTR == 1].mean(axis=1)

    assert mu1 > mu0

    t = (mu1 + mu0) / 2 + 0.2
    predictionsLDA = np.zeros((DTE.shape[1]))
    for i in range(DTE.shape[1]):
        x = vcol(DTE[:, i])
        xT = W[:, 0].T @ x
        if xT >= t:
            predictionsLDA[i] = 1

    accuracyLDA = np.sum(predictionsLDA == LTE) / DTE.shape[1]
    print(f"accuracyLDA: {accuracyLDA * 100:.2f}%\n")




if __name__ == '__main__':
    main(4, 2, center=False)
