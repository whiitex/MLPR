import numpy as np
from matplotlib import pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *
from src.mlpr_functions.PCA import *
from src.mlpr_functions.LDA import *
from src.mlpr_functions.logpdf_GAU_ND import *
from src.mlpr_functions.MVG import *


def main(m_PCA, m_LDA, applyPCA, applyLDA):
    D, L = load_data('./trainData.txt')
    classes = ['Counterfeit', 'Genuine']
    
    DM = D - compute_mu_C(D)[0]

    # Pair-wise scatter plots dependencies

    ########################################################################################################################
    # PCA - Principal Component Analysis

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # visualize_pairwise(DTR, LTR, np.array([0,1]), classes, a=0.09)

    # fig, axs = plt.subplots(1, 3, figsize=(15,4.5))
    # for j in range(3):
    #     axs[j].scatter(DTR[2*j, LTR==0], DTR[2*j+1, LTR==0], label=classes[0], marker='.', alpha=0.24)
    #     axs[j].scatter(DTR[2*j, LTR==1], DTR[2*j+1, LTR==1], label=classes[1], marker='.', alpha=0.24)


    #     axs[j].set_title(f"Feature {2*j+1} x Feature {2*(j+1)}", fontsize=12)
    #     axs[j].legend(fontsize=10)

    # fig.tight_layout(pad=3)
    # # plt.show()
    # plt.savefig('latex/images/feat_12_34_56.pdf', format='pdf')


    # m_PCA = 6 # 4/5 should be good
    print(f"m_PCA: {m_PCA}")
    P_PCA, DTR_PCA = PCA(DTR, m_PCA)
    DTE_PCA = P_PCA.T @ DTE

    if applyPCA:
        DTR = apply_pca(P_PCA, DTR)
        DTE = apply_pca(P_PCA, DTE)

    # visualize_pairwise(DTR_PCA, LTR, [0,1], classes, a=0.1, bins=40)

    # fig, axs = plt.subplots(2, 3, figsize=(16,9))
    # matrix = P_PCA.T @ DTR
    # for i in range(2):
    #     for j in range(3):
    #         for cls in range(len(classes)):
    #             axs[i][j].hist(matrix[i*3 + j, LTR==cls], density=1, bins=35, label=classes[cls], alpha=0.65)
    #         axs[i][j].set_title(f"Feature {i}", fontsize=10)
    #         axs[i][j].legend(fontsize=8)

    # fig.tight_layout(pad=3)
    # # plt.show()
    # # plt.savefig('latex/images/PCA_6.pdf', format='pdf')


    ########################################################################################################################
    # LDA - Linear Discriminant Analysis
    
    # m_LDA = 3
    print(f"m_LDA: {m_LDA}\n")
    W, DTR_LDA = LDA(DTR, LTR, [0,1], min(m_PCA, m_LDA), True)
    DTE_LDA = W.T @ DTE

    # PCA vs LDA
    # fig, axs = plt.subplots(1, 2, figsize=(10.6,5))
    # for cls in range(len(classes)):
    #     axs[0].hist(DTR_PCA[0, LTR==cls], density=1, bins=45, label=classes[cls], alpha=0.65)
    # axs[0].set_title(f"PCA", fontsize=12)
    # axs[0].legend(fontsize=8)
 
    # for cls in range(len(classes)):
    #     axs[1].hist(DTR_LDA[0, LTR==cls], density=0, bins=45, label=classes[cls], alpha=0.65)
    # axs[1].set_title(f"LDA", fontsize=12)
    # axs[1].legend(fontsize=8)
 
    # fig.tight_layout(pad=3)
    # plt.show()
    # plt.savefig('latex/images/PCAvsLDA.pdf', format='pdf')

    # visualize_pairwise(vrow(DTR_LDA[0, :]), LTR, [0,1], classes, a=0.1) 

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

    if applyLDA:
        DTR = apply_lda(W, DTR)
        DTE = apply_lda(W, DTE)

    ########################################################################################################################
    # Gaussian Multivariate density fitting

    # fig, axs = plt.subplots(4, 3, figsize=(20,16))
    # for c in range(2):
        # Dc = np.sort(DTR[:, LTR == c], axis=1)
        # mu = Dc.mean(axis=1).reshape(Dc.shape[0], 1)
        # for i in range(DTR.shape[0]):
            # row = Dc[i,:].reshape(1, Dc.shape[1])
            # Sigma = (row - mu[i]) @ (row - mu[i]).T / row.shape[1]
            # Sigma = np.ones((1,1)) * Sigma
            # if c == 0:
                # axs[c*2 + i//3][i%3].hist(row.ravel(), label=classes[c], density=1, bins=60, alpha=.8)
                # axs[c*2 + i//3][i%3].plot(row.ravel(), np.exp(logpdf_GAU_ND(row, mu[i], Sigma)), linewidth=1.75)
            # else:
                # axs[c*2 + i//3][i%3].plot(row.ravel(), np.exp(logpdf_GAU_ND(row, mu[i], Sigma)), linewidth=1.75)
                # axs[c*2 + i//3][i%3].hist(row.ravel(), label=classes[c], density=1, bins=60, alpha=.8)

            # axs[c*2 + i//3][i%3].set_title(f"Feature {i+1}", fontsize=18)
            # axs[c*2 + i//3][i%3].legend(fontsize=12.5)

    # fig.tight_layout(pad=3)
    # plt.savefig('latex/images/gaussian_fitting.pdf', format='pdf')
    # plt.show()


    ########################################################################################################################
    # MVG model classification

    binaryMVG("accuracyMVG", DTR, LTR, DTE, LTE, classes)    


    ########################################################################################################################
    # MVG TIED model classification

    binaryMVGTied("accuracyMVG_Tied", DTR, LTR, DTE, LTE, classes)


    ########################################################################################################################
    # MVG Naive Bayes model classification

    # we can achieve the naive bayes model just by cancelling the off-diagonal elements of the Sigma matrix
    binaryMVGnaive("accuracyMVG_Naive", DTR, LTR, DTE, LTE, classes)


    ########################################################################################################################
    # Pearson's correlation coefficient
    # Pearson's correlation coefficient
    S0 = np.cov(DTR[:, LTR == 0])
    S1 = np.cov(DTR[:, LTR == 1])

    try:
        PCS0 = S0 / ((S0.diagonal()**0.5).reshape(S0.shape[0],1) * (S0.diagonal()**0.5).reshape(1,S0.shape[0]))
        PCS1 = S1 / ((S1.diagonal()**0.5).reshape(S1.shape[0],1) * (S1.diagonal()**0.5).reshape(1,S1.shape[0]))
    except: pass

    # print("\nPearson's correlation coefficient class 0:")
    # print(np.round(PCS0, 3))
    # print("\nPearson's correlation coefficient class 1:")
    # print(np.round(PCS1, 3))

    ########################################################################################################################
    # MVG using only first 4 features

    print()
    binaryMVG("accuracyMVG_4", DTR[:4], LTR, DTE[:4], LTE, classes)
    binaryMVGTied("accuracyMVG_Tied_4", DTR[:4], LTR, DTE[:4], LTE, classes)
    binaryMVGnaive("accuracyMVG_Naive_4", DTR[:4], LTR, DTE[:4], LTE, classes)

    print()
    binaryMVG("accuracyMVG_12", DTR[:2], LTR, DTE[:2], LTE, classes)
    binaryMVGTied("accuracyMVG_Tied_12", DTR[:4], LTR, DTE[:4], LTE, classes)

    binaryMVG("accuracyMVG_34", DTR[2:4], LTR, DTE[2:4], LTE, classes)
    binaryMVGTied("accuracyMVG_Tied_34", DTR[2:4], LTR, DTE[2:4], LTE, classes)

    binaryMVG("accuracyMVG_56", DTR[4:], LTR, DTE[4:], LTE, classes)
    binaryMVGTied("accuracyMVG_Tied_56", DTR[4:], LTR, DTE[4:], LTE, classes)


########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    main(m_PCA=6, m_LDA=1, applyPCA=False, applyLDA=False)
