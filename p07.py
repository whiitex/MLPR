import numpy as np
import matplotlib.pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *
from src.mlpr_functions.PCA import *
from src.mlpr_functions.LDA import *
from src.mlpr_functions.logpdf_GAU_ND import *
from src.mlpr_functions.MVG import *
from src.mlpr_functions.BayesRisk import *


def main(m_PCA, m_LDA, applyPCA, applyLDA, center):
    D, L = load_data('data_et_checks/trainData.txt')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    DEVAL, LEVAL = load_data('data_et_checks/evalData.txt')
    classes = ['Counterfeit', 'Genuine']

    if center:
        DTR = DTR - compute_mu_C(DTR)[0] # centering data
        DTE = DTE - compute_mu_C(DTR)[0] # centering data
    
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
    # MVG classifier for different applications

    def compute_pT(pT, Cfn, Cfp):
        return (pT * Cfn) / (pT * Cfn + (1 - pT) * Cfp)
    
    colors = ['tab:orange', 'tab:green', 'tab:blue']
    for pT, Cfn, Cfp in [(.5,1,1), (.9,1,1), (.1,1,1), (.5,1,9), (.5,9,1)]:
        PT = compute_pT(pT, Cfn, Cfp)

        # MVG
        llrMVG = binaryMVG("accuracyMVG", DTR, LTR, DTE, LTE, classes, PT)
        _, dcfMVG = compute_bayes_risk_binary(llrMVG, LTE, pT, Cfn, Cfp)
        mindcfMVG, _ = compute_minDCF_binary(llrMVG, LTE, pT, Cfn, Cfp)
        print("MVG\t", end='')
        print(f'-> DCF: {dcfMVG:.3f}, DCFmin: {mindcfMVG:.3f}\n')

        x, y, z = plot_Bayes_errorXXX(llrMVG, LTE, -4, 4, 100)
        plt.plot(x, y, label='MVG DCF', color=colors[0])
        plt.plot(x, z, label='MVG minDCF', color=colors[0], linestyle='dashed')


        # MVG Tied
        llrMVGTied = binaryMVGTied("accuracyMVG_Tied", DTR, LTR, DTE, LTE, classes, pT=PT)
        _, dcfMVGTied = compute_bayes_risk_binary(llrMVGTied, LTE, pT, Cfn, Cfp)
        mindcfMVGTied, _ = compute_minDCF_binary(llrMVGTied, LTE, pT, Cfn, Cfp)
        print("MVGTied\t", end='')
        print(f'-> DCF: {dcfMVGTied:.3f}, DCFmin: {mindcfMVGTied:.3f}\n')

        x, y, z = plot_Bayes_errorXXX(llrMVGTied, LTE, -4, 4, 100)
        plt.plot(x, y, label='MVGTied DCF', color=colors[1])
        plt.plot(x, z, label='MVGTied minDCF', color=colors[1], linestyle='dashed')


        # MVG Naive
        llrMVGnaive = binaryMVGnaive("accuracyMVG_Naive", DTR, LTR, DTE, LTE, classes, pT=PT)
        _, dcfMVGnaive = compute_bayes_risk_binary(llrMVGnaive, LTE, pT, Cfn, Cfp)
        mindcfMVGnaive, _ = compute_minDCF_binary(llrMVGnaive, LTE, pT, Cfn, Cfp)
        print("MVGnaive\t", end='')
        print(f'-> DCF: {dcfMVGnaive:.3f}, DCFmin: {mindcfMVGnaive:.3f}\n')

        x, y, z = plot_Bayes_errorXXX(llrMVGnaive, LTE, -4, 4, 100)
        plt.plot(x, y, label='MVGnaive DCF', color=colors[2])
        plt.plot(x, z, label='MVGnaive minDCF', color=colors[2], linestyle='dashed')

        plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
        plt.ylabel("DCF", fontsize=12)
        plt.xlim(-4, 4)
        plt.ylim(0, 1.19)
        plt.axvline(x=2.1972, color='black', linestyle='--')
        plt.axvline(x=-2.1972, color='black', linestyle='--')
        plt.title(rf"Bayes error rate - $\pi_T = {pT}, C_n = {Cfn}, C_p = {Cfp}$", fontsize=12)
        plt.legend()
        plt.show()

    


if __name__ == '__main__':
    main(m_PCA=5, m_LDA=4, applyPCA=False, applyLDA=False, center=False)
