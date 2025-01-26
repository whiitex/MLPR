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
    # MVG model classification

    plt.figure(figsize=(8,6), tight_layout=True)
    pT, Cfn, Cfp = 0.1, 1, 1
    llr = binaryMVG("accuracyMVG", DTR, LTR, DTE, LTE, classes, pT)

    _, dcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    # plot_Bayes_error(llr, LTE, -4, 4, 100)
    print(f'-> DCF: {dcf:.3f}, DCFmin: {mindcf:.3f}\n')

    x, y, z = plot_Bayes_errorXXX(llr, LTE, -4, 4, 100)
    plt.plot(x, y, label='MVG DCF', color='r')
    plt.plot(x, z, label='MVG minDCF', color='r', linestyle='dashed')



    #########################################################################
    # MVG TIED model classification

    llr = binaryMVGTied("accuracyMVG_Tied", DTR, LTR, DTE, LTE, classes, pT=pT)

    _, dcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    # plot_Bayes_error(llr, LTE, -4, 4, 100)
    print(f'-> DCF: {dcf:.3f}, DCFmin: {mindcf:.3f}\n')

    x, y, z = plot_Bayes_errorXXX(llr, LTE, -4, 4, 100)
    plt.plot(x, y, label='Tied DCF', color='g')
    plt.plot(x, z, label='Tied minDCF', color='g', linestyle='dashed')

 

    #########################################################################
    # MVG Naive Bayes model classification

    # we can achieve the naive bayes model just by cancelling the off-diagonal elements of the Sigma matrix

    llr = binaryMVGnaive("accuracyMVG_Naive", DTR, LTR, DTE, LTE, classes)

    _, dcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    # plot_Bayes_error(llr, LTE, -4, 4, 100)
    print(f'-> DCF: {dcf:.3f}, DCFmin: {mindcf:.3f}')

    x, y, z = plot_Bayes_errorXXX(llr, LTE, -4, 4, 100)
    plt.plot(x, y, label='Tied DCF', color='b')
    plt.plot(x, z, label='Tied minDCF', color='b', linestyle='dashed')

    plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    plt.ylabel("DCF", fontsize=12)
    plt.xlim(-4, 4)
    plt.ylim(0, 1.19)
    plt.axvline(x=2.1972, color='black', linestyle='--')
    plt.axvline(x=-2.1972, color='black', linestyle='--')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title("Bayes error rate", fontsize=12)
    plt.legend()
    plt.show()
    # plt.savefig('latex/images/MVG_bayesdecision.pdf', format='pdf')



    #########################################################################
    # Pearson's correlation coefficient
    S0 = np.cov(DTR[:, LTR == 0])
    S1 = np.cov(DTR[:, LTR == 1])

    try:
        PCS0 = S0 / ((S0.diagonal()**0.5).reshape(S0.shape[0],1) * (S0.diagonal()**0.5).reshape(1,S0.shape[0]))
        PCS1 = S1 / ((S1.diagonal()**0.5).reshape(S1.shape[0],1) * (S1.diagonal()**0.5).reshape(1,S1.shape[0]))
    except: pass

    print("\nPearson's correlation coefficient class 0:")
    print(np.round(PCS0, 3))
    print("\nPearson's correlation coefficient class 1:")
    print(np.round(PCS1, 3))



    #########################################################################
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

    print()
    


if __name__ == '__main__':
    main(m_PCA=5, m_LDA=4, applyPCA=False, applyLDA=False, center=False)
