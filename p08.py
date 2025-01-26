import numpy as np
import matplotlib.pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *
from src.mlpr_functions.PCA import *
from src.mlpr_functions.LDA import *
from src.mlpr_functions.BayesRisk import *
from src.mlpr_functions.LogisticRegression import *


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
    # Logistic Regression

    pT, Cfn, Cfp = 0.1, 1, 1
    x, ydcf, ymindcf = [], [], []
    lamd = 0
    minErrLogReg = None
    for l in np.logspace(-4, 3, 1000):
        w, b = trainLogRegBinary(DTR, LTR, l)
        sVal = w.T @ DTE + b
        pEmp = (LTR == 1).sum() / LTR.size
        sValLlr = sVal - np.log(pEmp / (1-pEmp))
        RES = (sVal > 0) * 1
        accuracyLogReg = np.mean(RES == LTE)
        errorLogReg = 1 - accuracyLogReg
        if minErrLogReg is None or errorLogReg < minErrLogReg:
            minErrLogReg = errorLogReg
            lamd = l
        _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
        mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)
        x.append(l)
        ydcf.append(dcf)
        ymindcf.append(mindcf)
    
    plt.plot(x, ydcf, label='actDCF')
    plt.plot(x, ymindcf, label='minDCF')
    plt.xscale('log', base=10)
    plt.xlabel('$\lambda$')
    plt.ylabel('DCF')
    plt.legend()
    plt.show()
    # plt.savefig('latex/images/logreg_plot_dcf_lambda_fewsamples.pdf', format='pdf')


    pT, Cfn, Cfp = 0.1, 1, 1
    lamd = 0.03 # best l = 0.03 -> acc = 90.80%
    w, b = trainLogRegBinary(DTR, LTR, lamd)
    sVal = w.T @ DTE + b
    pEmp = (LTR == 1).sum() / LTR.size
    sValLlr = sVal - np.log(pEmp / (1-pEmp))
    RES = (sVal > 0) * 1
    accuracyLogReg = np.mean(RES == LTE)
    errorLogReg = 1 - accuracyLogReg
    _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)

    print(f"LinearLR - \u03BB: {lamd:.0e} - accuracyLogReg: {accuracyLogReg * 100:.2f}% - DCF {dcf:.3f} - minDCF {mindcf:.3f}")



    #########################################################################
    # Weighted Logistic Regression

    lamd, prior = 0, 0
    x, ydcf, ymindcf = [], [], []
    minErrLogReg = None
    space = np.linspace(0, 1, 5, endpoint=False)[1 ::].tolist()
    space.append(0.5)
    plt.figure(figsize=(8,5), tight_layout=True)
    for p in space:
        for l in np.logspace(-4, 3, 200):
            w, b = trainWeightedLogRegBinary(DTR, LTR, l, p)
            sVal = w.T @ DTE + b
            sValLlr = sVal - np.log(p / (1-p))
            RES = (sVal > 0) * 1
            accuracyLogReg = np.mean(RES == LTE)
            if minErrLogReg is None or 1 - accuracyLogReg < minErrLogReg:
                minErrLogReg = 1 - accuracyLogReg
                lamd, prior = l, p
            _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
            mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)
            x.append(l)
            ydcf.append(dcf)
            ymindcf.append(mindcf)
        # print(f"Caricamento: {p * 100:.2f}%")

        if (p != 0.5):
            plt.plot(x, ydcf, label=f'actDCF_p{p:.2f}')
            plt.plot(x, ymindcf, label=f'minDCF_p{p:.2f}')
        else:
            plt.plot(x, ydcf, label=f'actDCF', linestyle='dashed')
            plt.plot(x, ymindcf, label=f'minDCF', linestyle='dashed')

        plt.xscale('log', base=10)
        plt.xlabel('$\lambda$')
        plt.ylabel('DCF')
        print(f"p: {p:.2f} - DCF: {min(ydcf):.3f} - minDCF: {min(ymindcf):.3f}")
        ydcf.clear()
        ymindcf.clear()
        x.clear()
    plt.legend()
    plt.show()
    # plt.savefig('latex/images/logreg_weighted_plot_dcf_lambda.pdf', format='pdf')


    #########################################################################
    # Quadratic Logistic Regression

    # mu, _ = compute_mu_C(DTR)
    # DTR, DTE = DTR - mu, DTE - mu # centering
    pT, Cfn, Cfp = 0.1, 1, 1
    lamd = 2e-2
    x, ydcf, ymindcf = [], [], []
    minErrLogReg = None
    for l in np.logspace(-5, 3, 100):
    # for l in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        w, b = trainQuadraticLogRegBinary(DTR, LTR, l)
        phi_DTE = phi_x(DTE)
        sVal = w.T @ phi_DTE + b
        pEmp = (LTR == 1).sum() / LTR.size
        sValLlr = sVal - np.log(pEmp / (1-pEmp))
        RES = (sVal > 0) * 1
        accuracyLogReg = np.mean(RES == LTE)
        errorLogReg = 1 - accuracyLogReg
        if minErrLogReg is None or errorLogReg < minErrLogReg:
            minErrLogReg = errorLogReg
            lamd = l
        _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
        mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)
        x.append(l)
        ydcf.append(dcf)
        ymindcf.append(mindcf)
        print(f"\u03BB: {l:.1e} - accuracy: {accuracyLogReg * 100:.2f}% - DCF: {dcf:.3f} - minDCF: {mindcf:.3f}")

    
    plt.plot(x, ydcf, label='actDCF')
    plt.plot(x, ymindcf, label='minDCF')
    plt.xscale('log', base=10)
    plt.xlabel('$\lambda$')
    plt.ylabel('DCF')
    plt.legend()
    plt.show()
    # plt.savefig('latex/images/quadlogreg_plot_dcf_lambda.pdf', format='pdf')


    # Quadratic Logistic Regression
    w, b = trainQuadraticLogRegBinary(DTR, LTR, lamd)
    phi_DTE = phi_x(DTE)
    sVal = w.T @ phi_DTE + b
    pEmp = (LTR == 1).sum() / LTR.size
    sValLlr = sVal - np.log(pEmp / (1-pEmp))
    RES = (sValLlr > 0) * 1
    accuracyLogReg, errorLogReg = np.mean(RES == LTE), 1 - np.mean(RES == LTE)
    _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)

    print(f"QuadLR - \u03BB: {lamd:.0e} - accuracyLogReg: {accuracyLogReg * 100:.2f}% - DCF {dcf:.3f} - minDCF {mindcf:.3f}\n")

    llrQuadraticLogReg = sValLlr
    llrEvalQuadraticLogReg = w.T @ phi_x(DEVAL) + b - pEmp
    # np.save('models/llrQuadraticLogReg.npy', llrQuadraticLogReg)
    # np.save('models/llrEvalQuadraticLogReg.npy', llrEvalQuadraticLogReg)



if __name__ == '__main__':
    main(m_PCA=5, m_LDA=4, applyPCA=False, applyLDA=False, center=False)
