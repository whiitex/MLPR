import numpy as np
import matplotlib.pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *
from src.mlpr_functions.PCA import *
from src.mlpr_functions.LDA import *
from src.mlpr_functions.BayesRisk import *
from src.mlpr_functions.SVM import *


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
    # SVM

    # DTR = DTR - compute_mu_C(DTR)[0] # centering data
    # DTE = DTE - compute_mu_C(DTR)[0] # centering data
    x, ydcf, ymindcf = [], [], []
    pT, Cfn, Cfp = 0.1, 1, 1
    for K in np.linspace(1, 10, 1):
        for C in np.logspace(-5, 0, 11):
            w, b, res = train_dual_SVM_linear(DTR, LTR, C, K)
            SVAL = (vrow(w) @ DTE + b).ravel()
            PVAL = (SVAL > 0) * 1
            acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
            mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
            _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
            print(f"K: {K:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}")
            x.append(C)
            ydcf.append(actdcf)
            ymindcf.append(mindcf)
    plt.plot(x, ydcf, label='actDCF')
    plt.plot(x, ymindcf, label='minDCF')
    plt.xscale('log', base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.title('Linear SVM - actDCF and minDCF')
    plt.show()
    # plt.savefig('latex/images/linear_svm_plot_C_dcf.pdf', format='pdf')


    #########################################################################
    # SVM Polynomial Kernel

    x, ydcf, ymindcf = [], [], []
    pT, Cfn, Cfp = 0.1, 1, 1
    for eps in np.linspace(1,10,1):
        for C in np.logspace(-5, 0, 11):
            fScore, _ = train_dual_SVM_kernel(DTR, LTR, C, polynomialKernel(4, 1), eps)
            SVAL = fScore(DTE)
            PVAL = (SVAL > 0) * 1
            acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
            mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
            _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
            print(f"eps: {eps:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}")
            x.append(C)
            ydcf.append(actdcf)
            ymindcf.append(mindcf)
    plt.plot(x, ydcf, label='actDCF')
    plt.plot(x, ymindcf, label='minDCF')
    plt.xscale('log', base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.title('Polynomial kernel SVM - actDCF and minDCF')
    plt.show()
    # plt.savefig('latex/images/poly4_svm_plot_C_dcf.pdf', format='pdf')

    # SVM Polynomial
    deg, K, C = 4, 1.0, 3.16e-3
    fScore, _ = train_dual_SVM_kernel(DTR, LTR, C, polynomialKernel(deg, 1), K)
    SVAL = fScore(DTE)
    PVAL = (SVAL > 0) * 1
    acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
    mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
    _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
    print(f"eps: {K:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}")

    # llrPoly4SVM = SVAL
    # llrEvalPoly4SVM = fScore(DEVAL)
    # np.save('models/llrPoly4SVM.npy', llrPoly4SVM)
    # np.save('models/llrEvalPoly4SVM.npy', llrEvalPoly4SVM)



    #########################################################################
    # SVM RBF Kernel

    pT, Cfn, Cfp = 0.1, 1, 1
    x, ydcf, ymindcf = [], [], []
    pT, Cfn, Cfp = 0.1, 1, 1
    i = -4
    colors = ['b', 'g', 'r', 'c']
    for g in [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]:
        for C in np.logspace(-3, 2, 11):
            fScore, _ = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(g), 1)
            SVAL = fScore(DTE)
            PVAL = (SVAL > 0) * 1
            acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
            mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
            _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
            print(f"gam: {g:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}")
            x.append(C)
            ydcf.append(actdcf)
            ymindcf.append(mindcf)
        plt.plot(x, ydcf, label=f'actDCF_e{i}', color=colors[i+4])
        plt.plot(x, ymindcf, label=f'minDCF_e{i}', linestyle='dashed', color=colors[i+4])
        i += 1
        x.clear()
        ydcf.clear()
        ymindcf.clear()
    plt.xscale('log', base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.title('RBF kernel SVM - actDCF and minDCF')
    plt.show()
    # plt.savefig('latex/images/rbf_svm_plot_C_dcf.pdf', format='pdf')


    # SVM RBF
    g, C = np.exp(-2), 3.16e1
    fScore, _ = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(g), 1)
    SVAL = fScore(DTE)
    PVAL = (SVAL > 0) * 1
    acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
    mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
    _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
    print(f"gam: {g:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}\n")

    # llrRBFSVM = SVAL
    # llrEvalRBFSVM = fScore(DEVAL)
    # np.save('models/llrRBFSVM.npy', llrRBFSVM)
    # np.save('models/llrEvalRBFSVM.npy', llrEvalRBFSVM)



if __name__ == '__main__':
    main(m_PCA=5, m_LDA=4, applyPCA=False, applyLDA=False, center=False)
