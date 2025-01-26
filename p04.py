import numpy as np
import matplotlib.pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *
from src.mlpr_functions.PCA import *
from src.mlpr_functions.LDA import *
from src.mlpr_functions.logpdf_GAU_ND import *


def main(m_PCA, m_LDA, applyPCA, applyLDA, center):
    D, L = load_data('data_et_checks/trainData.txt')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    DEVAL, LEVAL = load_data('data_et_checks/evalData.txt')
    classes = ['Counterfeit', 'Genuine']

    if center:
        DTR = DTR - compute_mu_C(DTR)[0] # cetering data
        DTE = DTE - compute_mu_C(DTR)[0] # cetering data
    
    print(f"m_PCA: {m_PCA}, applyPCA: {applyPCA}")
    P_PCA, DTR_PCA = PCA(DTR, m_PCA)
    DTE_PCA = P_PCA.T @ DTE
    if applyPCA:
        DTR = apply_pca(P_PCA, DTR)
        DTE = apply_pca(P_PCA, DTE)

    print(f"m_LDA: {m_LDA}, applyLDA: {applyLDA}\n")
    W, DTR_LDA = LDA(DTR, LTR, [0,1], min(m_PCA, m_LDA), True)
    DTE_LDA = W.T @ DTE
    if applyLDA:
        DTR = apply_lda(W, DTR)
        DTE = apply_lda(W, DTE)
    
    #########################################################################
    # Gaussian Multivariate density fitting

    fig, axs = plt.subplots(4, 3, figsize=(20,16))
    for c in range(2):
        Dc = np.sort(DTR[:, LTR == c], axis=1)
        mu = Dc.mean(axis=1).reshape(Dc.shape[0], 1)
        for i in range(DTR.shape[0]):
            row = Dc[i,:].reshape(1, Dc.shape[1])
            Sigma = (row - mu[i]) @ (row - mu[i]).T / row.shape[1]
            Sigma = np.ones((1,1)) * Sigma
            if c == 0:
                axs[c*2 + i//3][i%3].hist(row.ravel(), label=classes[c], density=1, bins=60, alpha=.8)
                axs[c*2 + i//3][i%3].plot(row.ravel(), np.exp(logpdf_GAU_ND(row, mu[i], Sigma)), linewidth=1.75)
            else:
                axs[c*2 + i//3][i%3].plot(row.ravel(), np.exp(logpdf_GAU_ND(row, mu[i], Sigma)), linewidth=1.75)
                axs[c*2 + i//3][i%3].hist(row.ravel(), label=classes[c], density=1, bins=60, alpha=.8)

            axs[c*2 + i//3][i%3].set_title(f"Feature {i+1}", fontsize=18)
            axs[c*2 + i//3][i%3].legend(fontsize=12.5)

    fig.tight_layout(pad=3)
    plt.show()
    # plt.savefig('latex/images/gaussian_fitting.pdf', format='pdf')

    


if __name__ == '__main__':
    main(m_PCA=5, m_LDA=4, applyPCA=False, applyLDA=False, center=False)
