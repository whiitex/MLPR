import numpy as np
import matplotlib.pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *
from src.mlpr_functions.PCA import *
from src.mlpr_functions.LDA import *
from src.mlpr_functions.logpdf_GAU_ND import *
from src.mlpr_functions.MVG import *
from src.mlpr_functions.BayesRisk import *
from src.mlpr_functions.LogisticRegression import *
from src.mlpr_functions.SVM import *
from src.mlpr_functions.GMM import *


def main(m_PCA, m_LDA, applyPCA, applyLDA, center):
    D, L = load_data('data_et_checks/trainData.txt')
    DEVAL, LEVAL = load_data('data_et_checks/evalData.txt')
    classes = ['Counterfeit', 'Genuine']
    
    DM = D - compute_mu_C(D)[0]

    # Pair-wise scatter plots dependencies


    ########################################################################################################################
    # PCA - Principal Component Analysis

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    if center:
        DTR = DTR - compute_mu_C(DTR)[0] # cetering data
        DTE = DTE - compute_mu_C(DTR)[0] # cetering data

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
    print(f"m_PCA: {m_PCA}, applyPCA: {applyPCA}")
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
    # plt.show()
    # # plt.savefig('latex/images/PCA_6.pdf', format='pdf')


    ########################################################################################################################
    # LDA - Linear Discriminant Analysis
    
    # m_LDA = 3
    print(f"m_LDA: {m_LDA}, applyLDA: {applyLDA}\n")
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

    plt.figure(figsize=(8,6), tight_layout=True)
    pT, Cfn, Cfp = 0.1, 1, 1
    llr = binaryMVG("accuracyMVG", DTR, LTR, DTE, LTE, classes, pT)

    _, dcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    # plot_Bayes_error(llr, LTE, -4, 4, 100)
    print(f'-> DCF: {dcf:.3f}, DCFmin: {mindcf:.3f}\n')

    # x, y, z = plot_Bayes_errorXXX(llr, LTE, -4, 4, 100)
    # plt.plot(x, y, label='MVG DCF', color='r')
    # plt.plot(x, z, label='MVG minDCF', color='r', linestyle='dashed')



    ########################################################################################################################
    # MVG TIED model classification

    llr = binaryMVGTied("accuracyMVG_Tied", DTR, LTR, DTE, LTE, classes, pT=pT)

    _, dcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    # plot_Bayes_error(llr, LTE, -4, 4, 100)
    print(f'-> DCF: {dcf:.3f}, DCFmin: {mindcf:.3f}\n')

    # x, y, z = plot_Bayes_errorXXX(llr, LTE, -4, 4, 100)
    # plt.plot(x, y, label='Tied DCF', color='g')
    # plt.plot(x, z, label='Tied minDCF', color='g', linestyle='dashed')

 

    ########################################################################################################################
    # MVG Naive Bayes model classification

    # we can achieve the naive bayes model just by cancelling the off-diagonal elements of the Sigma matrix
    
    llr = binaryMVGnaive("accuracyMVG_Naive", DTR, LTR, DTE, LTE, classes)

    _, dcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    # plot_Bayes_error(llr, LTE, -4, 4, 100)
    print(f'-> DCF: {dcf:.3f}, DCFmin: {mindcf:.3f}')

    # x, y, z = plot_Bayes_errorXXX(llr, LTE, -4, 4, 100)
    # plt.plot(x, y, label='Tied DCF', color='b')
    # plt.plot(x, z, label='Tied minDCF', color='b', linestyle='dashed')

    # plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    # plt.ylabel("DCF", fontsize=12)
    # plt.xlim(-4, 4)
    # plt.ylim(0, 1.19)
    # plt.axvline(x=2.1972, color='black', linestyle='--')
    # plt.axvline(x=-2.1972, color='black', linestyle='--')
    # plt.axvline(x=0, color='black', linestyle='--')
    # plt.legend()
    # plt.savefig('latex/images/MVG_bayesdecision.pdf', format='pdf')



    ########################################################################################################################
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

    print()


    ########################################################################################################################
    # Logistic Regression

    # pT, Cfn, Cfp = 0.1, 1, 1
    # x, ydcf, ymindcf = [], [], []
    # lamd = 0
    # minErrLogReg = None
    # for l in np.logspace(-4, 3, 1000):
    #     w, b = trainLogRegBinary(DTR, LTR, l)
    #     sVal = w.T @ DTE + b
    #     pEmp = (LTR == 1).sum() / LTR.size
    #     sValLlr = sVal - np.log(pEmp / (1-pEmp))
    #     RES = (sVal > 0) * 1
    #     accuracyLogReg = np.mean(RES == LTE)
    #     errorLogReg = 1 - accuracyLogReg
    #     if minErrLogReg is None or errorLogReg < minErrLogReg:
    #         minErrLogReg = errorLogReg
    #         lamd = l
    #     _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
    #     mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)
    #     x.append(l)
    #     ydcf.append(dcf)
    #     ymindcf.append(mindcf)
    
    # plt.plot(x, ydcf, label='actDCF')
    # plt.plot(x, ymindcf, label='minDCF')
    # plt.xscale('log', base=10)
    # plt.xlabel('$\lambda$')
    # plt.ylabel('DCF')
    # plt.legend()
    # plt.show()
    # plt.savefig('latex/images/logreg_plot_dcf_lambda_fewsamples.pdf', format='pdf')


    pT, Cfn, Cfp = 0.1, 1, 1
    lamd = 0.03 # best l = 0.03 -> acc = 90.80%
    w, b = trainLogRegBinary(DTR, LTR, lamd)
    sVal = w.T @ DTE + b
    pEmp = (LTR == 1).sum() / LTR.size
    sValLlr = sVal - np.log(pEmp / (1-pEmp))
    RES = (sVal > 0) * 1
    accuracyLogReg = np.mean(RES == LTE)
    # errorLogReg = 1 - accuracyLogReg
    _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)

    print(f"LinearLR - \u03BB: {lamd:.0e} - accuracyLogReg: {accuracyLogReg * 100:.2f}% - DCF {dcf:.3f} - minDCF {mindcf:.3f}")



    ########################################################################################################################
    # Weighted Logistic Regression

    # lamd = 0
    # prior = 0
    # x, ydcf, ymindcf = [], [], []
    # minErrLogReg = None
    # space = np.linspace(0, 1, 5, endpoint=False)[1 ::].tolist()
    # space.append(0.5)
    # plt.figure(figsize=(8,5), tight_layout=True)
    # for p in space:
    #     for l in np.logspace(-4, 3, 200):
    #         w, b = trainWeightedLogRegBinary(DTR, LTR, l, p)
    #         sVal = w.T @ DTE + b
    #         sValLlr = sVal - np.log(p / (1-p))
    #         RES = (sVal > 0) * 1
    #         accuracyLogReg = np.mean(RES == LTE)
    #         if minErrLogReg is None or 1 - accuracyLogReg < minErrLogReg:
    #             minErrLogReg = 1 - accuracyLogReg
    #             lamd, prior = l, p
    #         _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
    #         mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)
    #         x.append(l)
    #         ydcf.append(dcf)
    #         ymindcf.append(mindcf)
    #     # print(f"Caricamento: {p * 100:.2f}%")

    #     if (p != 0.5):
    #         plt.plot(x, ydcf, label=f'actDCF_p{p:.2f}')
    #         plt.plot(x, ymindcf, label=f'minDCF_p{p:.2f}')
    #     else:
    #         plt.plot(x, ydcf, label=f'actDCF', linestyle='dashed')
    #         plt.plot(x, ymindcf, label=f'minDCF', linestyle='dashed')

    #     plt.xscale('log', base=10)
    #     plt.xlabel('$\lambda$')
    #     plt.ylabel('DCF')
    #     print(f"p: {p:.2f} - DCF: {min(ydcf):.3f} - minDCF: {min(ymindcf):.3f}")
    #     ydcf.clear()
    #     ymindcf.clear()
    #     x.clear()
    # plt.legend()
    # plt.show()
    # plt.savefig('latex/images/logreg_weighted_plot_dcf_lambda.pdf', format='pdf')


    ########################################################################################################################
    # Quadratic Logistic Regression

    mu, _ = compute_mu_C(DTR)
    # DTR, DTE = DTR - mu, DTE - mu # centering
    pT, Cfn, Cfp = 0.1, 1, 1
    lamd = 2e-2
    # x, ydcf, ymindcf = [], [], []
    # minErrLogReg = None
    # for l in np.logspace(-5, 3, 100):
    # # for l in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    #     w, b = trainQuadraticLogRegBinary(DTR, LTR, l)
    #     phi_DTE = phi_x(DTE)
    #     sVal = w.T @ phi_DTE + b
    #     pEmp = (LTR == 1).sum() / LTR.size
    #     sValLlr = sVal - np.log(pEmp / (1-pEmp))
    #     RES = (sVal > 0) * 1
    #     accuracyLogReg = np.mean(RES == LTE)
    #     errorLogReg = 1 - accuracyLogReg
    #     if minErrLogReg is None or errorLogReg < minErrLogReg:
    #         minErrLogReg = errorLogReg
    #         lamd = l
    #     _, dcf = compute_bayes_risk_binary(sValLlr, LTE, pT, Cfn, Cfp)
    #     mindcf, _ = compute_minDCF_binary(sValLlr, LTE, pT, Cfn, Cfp)
        # x.append(l)
        # ydcf.append(dcf)
        # ymindcf.append(mindcf)
        # print(f"\u03BB: {l:.1e} - accuracy: {accuracyLogReg * 100:.2f}% - DCF: {dcf:.3f} - minDCF: {mindcf:.3f}")

    
    # plt.plot(x, ydcf, label='actDCF')
    # plt.plot(x, ymindcf, label='minDCF')
    # plt.xscale('log', base=10)
    # plt.xlabel('$\lambda$')
    # plt.ylabel('DCF')
    # plt.legend()
    # plt.show()
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

    # llrQuadraticLogReg = sValLlr
    # llrEvalQuadraticLogReg = w.T @ phi_x(DEVAL) + b - pEmp
    

    ########################################################################################################################
    # SVM

    # DTR = DTR - compute_mu_C(DTR)[0] # cetering data
    # DTE = DTE - compute_mu_C(DTR)[0] # cetering data
    # x, ydcf, ymindcf = [], [], []
    # pT, Cfn, Cfp = 0.1, 1, 1
    # for K in np.linspace(1, 10, 1):
    #     for C in np.logspace(-5, 0, 11):
    #         w, b, res = train_dual_SVM_linear(DTR, LTR, C, K)
    #         SVAL = (vrow(w) @ DTE + b).ravel()
    #         PVAL = (SVAL > 0) * 1
    #         acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
    #         mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
    #         _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
    #         print(f"K: {K:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}")
    #         x.append(C)
    #         ydcf.append(actdcf)
    #         ymindcf.append(mindcf)
    # plt.plot(x, ydcf, label='actDCF')
    # plt.plot(x, ymindcf, label='minDCF')
    # plt.xscale('log', base=10)
    # plt.xlabel('C')
    # plt.ylabel('DCF')
    # plt.legend()
    # plt.title('Linear SVM - actDCF and minDCF')
    # plt.show()
    # plt.savefig('latex/images/linear_svm_plot_C_dcf.pdf', format='pdf')


    ########################################################################################################################
    # SVM Polynomial Kernel

    # x, ydcf, ymindcf = [], [], []
    # pT, Cfn, Cfp = 0.1, 1, 1
    # for eps in np.linspace(1,10,1):
    #     for C in np.logspace(-5, 0, 11):
    #         fScore, _ = train_dual_SVM_kernel(DTR, LTR, C, polynomialKernel(4, 1), eps)
    #         SVAL = fScore(DTE)
    #         PVAL = (SVAL > 0) * 1
    #         acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
    #         mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
    #         _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
    #         print(f"eps: {eps:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}")
    #         x.append(C)
    #         ydcf.append(actdcf)
    #         ymindcf.append(mindcf)
    # plt.plot(x, ydcf, label='actDCF')
    # plt.plot(x, ymindcf, label='minDCF')
    # plt.xscale('log', base=10)
    # plt.xlabel('C')
    # plt.ylabel('DCF')
    # plt.legend()
    # plt.title('Polynomial kernel SVM - actDCF and minDCF')
    # # plt.show()
    # plt.savefig('latex/images/poly4_svm_plot_C_dcf.pdf', format='pdf')

    # SVM Polynomial
    # deg, K, C = 4, 1.0, 3.16e-3
    # fScore, _ = train_dual_SVM_kernel(DTR, LTR, C, polynomialKernel(deg, 1), K)
    # SVAL = fScore(DTE)
    # PVAL = (SVAL > 0) * 1
    # acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
    # mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
    # _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
    # print(f"eps: {K:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}")

    # llrPoly4SVM = SVAL
    # llrEvalPoly4SVM = fScore(DEVAL)


    ########################################################################################################################
    # SVM RBF Kernel

    # pT, Cfn, Cfp = 0.1, 1, 1
    # x, ydcf, ymindcf = [], [], []
    # pT, Cfn, Cfp = 0.1, 1, 1
    # i = -4
    # colors = ['b', 'g', 'r', 'c']
    # for g in [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]:
    #     for C in np.logspace(-3, 2, 11):
    #         fScore, _ = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(g), 1)
    #         SVAL = fScore(DTE)
    #         PVAL = (SVAL > 0) * 1
    #         acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
    #         mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
    #         _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
    #         print(f"gam: {g:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}")
    #         x.append(C)
    #         ydcf.append(actdcf)
    #         ymindcf.append(mindcf)
    #     plt.plot(x, ydcf, label=f'actDCF_e{i}', color=colors[i+4])
    #     plt.plot(x, ymindcf, label=f'minDCF_e{i}', linestyle='dashed', color=colors[i+4])
    #     i += 1
    #     x.clear()
    #     ydcf.clear()
    #     ymindcf.clear()
    # plt.xscale('log', base=10)
    # plt.xlabel('C')
    # plt.ylabel('DCF')
    # plt.legend()
    # plt.title('RBF kernel SVM - actDCF and minDCF')
    # plt.show()
    # plt.savefig('latex/images/rbf_svm_plot_C_dcf.pdf', format='pdf')


    # SVM RBF
    # g, C = np.exp(-2), 3.16e1
    # fScore, _ = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(g), 1)
    # SVAL = fScore(DTE)
    # PVAL = (SVAL > 0) * 1
    # acc, err = np.mean(PVAL == LTE), np.mean(PVAL != LTE)
    # mindcf, _ = compute_minDCF_binary(SVAL, LTE, pT, Cfn, Cfp)
    # _, actdcf = compute_bayes_risk_binary(SVAL, LTE, pT, Cfn, Cfp)
    # print(f"gam: {g:.1f} - C: {C:.2e} - Accuracy: {acc * 100:.2f}% - actDCF: {actdcf:.3f} - minDCF: {mindcf:.3f}\n")

    # llrRBFSVM = SVAL
    # llrEvalRBFSVM = fScore(DEVAL)



    ########################################################################################################################
    # GMM

    # pT, Cfn, Cfp = 0.1, 1, 1
    # for func in ['full', 'diagonal', 'tied']:
    #     for cl1 in [1,2,4,8,16,32]:
    #         for cl2 in [1,2,4,8,16,32]:
    #             try:
    #                 gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], cl1, covType=func, verbose=False)
    #                 gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], cl2, covType=func, verbose=False)
    #                 llr = classiy_GMM_binary(DTE, gmm0, gmm1, pT)
    #                 SVAL = (llr > 0) * 1
    #                 acc, err = np.mean(SVAL == LTE), 1 - np.mean(SVAL == LTE)
    #                 _, actdcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    #                 mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    #                 print(f'comp: [{cl1}, {cl2}] \t- func: {func[:4]} - acc: {acc * 100:.2f}% - dcf: {actdcf:.3f}, mindcf: {mindcf:.3f}')
    #             except: 
    #                 print(f'error comp: [{cl1}, {cl2}]')
    #                 continue


    # GMM diagonal
    # cl1, cl2, func = 4, 16, 'diagonal'
    # gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], cl1, covType=func, verbose=False)
    # gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], cl2, covType=func, verbose=False)
    # llr = classiy_GMM_binary(DTE, gmm0, gmm1, pT)
    # SVAL = (llr > 0) * 1
    # acc, err = np.mean(SVAL == LTE), 1 - np.mean(SVAL == LTE)
    # _, actdcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    # mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    # print(f'comp: [{cl1}, {cl2}] \t- func: {func[:4]} - acc: {acc * 100:.2f}% - dcf: {actdcf:.3f}, mindcf: {mindcf:.3f}')
    
    # llrGMMdiagonal = llr
    # llrEvalGMMdiagonal = classiy_GMM_binary(DEVAL, gmm0, gmm1, pT)


    ########################################################################################################################
    # Model comparison on different application

    # x,y,z = plot_Bayes_errorXXX(llrQuadraticLogReg, LTE, -4, 4, 50)
    # plt.plot(x, y, label='LogReg2 DCF', color='r')
    # plt.plot(x, z, label='LogReg2 minDCF', color='r', linestyle='dashed')

    # x,y,z = plot_Bayes_errorXXX(llrPoly4SVM, LTE, -4, 4, 50)
    # plt.plot(x, y, label='PolySVM4 DCF', color='b')
    # plt.plot(x, z, label='PolySVM4 minDCF', color='b', linestyle='dashed')

    # x,y,z = plot_Bayes_errorXXX(llrRBFSVM, LTE, -4, 4, 50)
    # plt.plot(x, y, label='RBFSVM DCF', color='m')
    # plt.plot(x, z, label='RBFSVM minDCF', color='m', linestyle='dashed')

    # x,y,z = plot_Bayes_errorXXX(llrGMMdiagonal, LTE, -4, 4, 50)
    # plt.plot(x, y, label='GMMdiag DCF', color='g')
    # plt.plot(x, z, label='GMMdiag minDCF', color='g', linestyle='dashed')

    # plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    # plt.ylabel("DCF", fontsize=12)
    # plt.xlim(-4, 4)
    # plt.ylim(0, 1.19)
    # plt.axvline(x=2.1972, color='black', linestyle='--')
    # plt.axvline(x=-2.1972, color='black', linestyle='--')
    # plt.axvline(x=0, color='black', linestyle='--')
    # plt.legend()
    # plt.title(r'Best models comparison -- Bayes error')
    # plt.show()
    # # plt.savefig('latex/images/best_model_bayes_error.pdf', format='pdf')


    ########################################################################################################################
    # LLR models saving

    # np.save('models/llrQuadraticLogReg.npy', llrQuadraticLogReg)
    # np.save('models/llrPoly4SVM.npy', llrPoly4SVM)
    # np.save('models/llrRBFSVM.npy', llrRBFSVM)
    # np.save('models/llrGMMdiagonal.npy', llrGMMdiagonal)

    # np.save('models/llrEvalQuadraticLogReg.npy', llrEvalQuadraticLogReg)
    # np.save('models/llrEvalPoly4SVM.npy', llrEvalPoly4SVM)
    # np.save('models/llrEvalRBFSVM.npy', llrEvalRBFSVM)
    # np.save('models/llrEvalGMMdiagonal.npy', llrEvalGMMdiagonal)

    scoresQuadLogReg = np.load('models/llrQuadraticLogReg.npy')
    scoresPoly4SVM = np.load('models/llrPoly4SVM.npy')
    scoresRBFSVM = np.load('models/llrRBFSVM.npy')
    scoresGMMdiag = np.load('models/llrGMMdiagonal.npy')

    scoresEvalQuadLogReg = np.load('models/llrEvalQuadraticLogReg.npy')
    scoresEvalPoly4SVM = np.load('models/llrEvalPoly4SVM.npy')
    scoresEvalRBFSVM = np.load('models/llrEvalRBFSVM.npy')
    scoresEvalGMMdiag = np.load('models/llrEvalGMMdiagonal.npy')


    ########################################################################################################################
    # K-fold cross validation on GMM diag

    pT, Cfn, Cfp = 0.1, 1, 1

    KFOLD = 10

    # K-fold on validation set
    print("K-fold on validation set")
    colors = ['tab:orange', 'tab:green', 'tab:blue']
    funcs = ['SVMpoly4', 'SVMRBF', 'GMMdiag']
    a = 0
    for score in [scoresQuadLogReg, scoresPoly4SVM, scoresRBFSVM, scoresGMMdiag]:
        folds, foldlab, SCAL, LCAL = [], [], [], []
        for i in range(KFOLD): 
            folds.append(score[i::KFOLD])
            foldlab.append(LTE[i::KFOLD])

        for i in range(KFOLD):
            train = np.hstack([folds[j] for j in range(KFOLD) if j != i])
            trainlab = np.hstack([foldlab[j] for j in range(KFOLD) if j != i])
            LCAL.append(foldlab[i])

            w, b = trainWeightedLogRegBinary(vrow(train), trainlab, 0, pT)
            res = w.T @ vrow(folds[i]) + b - np.log(pT / (1-pT))
            SCAL.append(res)
        
        
        SCAL = np.hstack(SCAL)
        LCAL = np.hstack(LCAL)
        res = (SCAL > 0) * 1
        acc = np.mean(res == LCAL)
        _, dcf = compute_bayes_risk_binary(SCAL, LCAL, pT, Cfn, Cfp)
        mindcf, _ = compute_minDCF_binary(SCAL, LCAL, pT, Cfn, Cfp)
        print(f"Accuracy: {acc * 100:.2f}% - DCF: {dcf:.3f} - minDCF: {mindcf:.3f}")

    #     x, y, z = plot_Bayes_errorXXX(SCAL, LCAL, -3.8, 3.8, 100)
    #     plt.plot(x, y, label=f'{funcs[a]} cal actDCF', color=colors[a])
    #     plt.plot(x, z, label=f'{funcs[a]} minDCF', color=colors[a], linestyle='dashed')
    #     x, y, _ = plot_Bayes_errorXXX(folds[i], foldlab[i], -3.8, 3.8, 100)
    #     plt.plot(x, y, label=f'{funcs[a]} raw actDCF', color=colors[a], linestyle='dotted')
    #     a += 1
    
    # plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    # plt.ylabel("DCF", fontsize=12)
    # # plt.xlim(-4, 4)
    # # plt.ylim(0, 1.19)
    # plt.axvline(x=2.1972, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=-2.1972, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    # plt.legend()
    # plt.title(r'Calibrated models comparison -- Bayes error')
    # plt.savefig('latex/images/calibrated_bayes_error_best_models_va.pdf', format='pdf')
    # plt.show()

    
    # K-fold on evaluation set
    print("\nK-fold on evaluation set")
    for score, evalscore in [(scoresQuadLogReg, scoresEvalQuadLogReg), (scoresPoly4SVM, scoresEvalPoly4SVM), (scoresRBFSVM, scoresEvalRBFSVM), (scoresGMMdiag, scoresEvalGMMdiag)]:
        
        w, b = trainWeightedLogRegBinary(vrow(score), LTE, 0, 0.1)
        res = w.T @ vrow(evalscore) + b - np.log(pT / (1-pT))
        _, dcf = compute_bayes_risk_binary(res, LEVAL, pT, Cfn, Cfp)
        mindcf, _ = compute_minDCF_binary(res, LEVAL, pT, Cfn, Cfp)
        res = (res > 0) * 1
        acc = np.mean(res == LEVAL)
        print(f"Accuracy: {acc * 100:.2f}% - DCF: {dcf:.3f} - minDCF: {mindcf:.3f}")

    # Bayes error plot of calibrated models
    # w, b = trainWeightedLogRegBinary(vrow(scoresGMMdiag), LTE, 0, pT)
    # res = w.T @ vrow(scoresEvalGMMdiag) + b - np.log(pT / (1-pT))
    # x, y, z = plot_Bayes_errorXXX(res, LEVAL, -4, 4, 100)
    # plt.plot(x, y, label='GMMdiag cal actDCF', color='tab:blue')
    # plt.plot(x, z, label='GMMdiag minDCF', color='tab:blue', linestyle='dashed')
    # x, y, _ = plot_Bayes_errorXXX(scoresEvalGMMdiag, LEVAL, -4, 4, 100)
    # plt.plot(x, y, label='GMMdiag raw actDCF', color='tab:blue', linestyle='dotted')
    
    # w, b = trainWeightedLogRegBinary(vrow(scoresPoly4SVM), LTE, 0, pT)
    # res = w.T @ vrow(scoresEvalPoly4SVM) + b - np.log(pT / (1-pT))
    # x, y, z = plot_Bayes_errorXXX(res, LEVAL, -4, 4, 100)
    # plt.plot(x, y, label='SVMpoly4 cal actDCF', color='tab:orange')
    # plt.plot(x, z, label='SVMpoly4 minDCF', color='tab:orange', linestyle='dashed')
    # x, y, _ = plot_Bayes_errorXXX(scoresEvalPoly4SVM, LEVAL, -4, 4, 100)
    # plt.plot(x, y, label='SVMpoly4 raw actDCF', color='tab:orange', linestyle='dotted')

    # w, b = trainWeightedLogRegBinary(vrow(scoresRBFSVM), LTE, 0, pT)
    # res = w.T @ vrow(scoresEvalRBFSVM) + b - np.log(pT / (1-pT))
    # x, y, z = plot_Bayes_errorXXX(res, LEVAL, -4, 4, 100)
    # plt.plot(x, y, label='RBFSVM cal actDCF', color='tab:green')
    # plt.plot(x, z, label='RBFSVM minDCF', color='tab:green', linestyle='dashed')
    # x, y, _ = plot_Bayes_errorXXX(scoresEvalRBFSVM, LEVAL, -4, 4, 100)
    # plt.plot(x, y, label='RBFSVM raw actDCF', color='tab:green', linestyle='dotted')

    # plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    # plt.ylabel("DCF", fontsize=12)
    # # plt.xlim(-4, 4)
    # # plt.ylim(0, 1.19)
    # plt.axvline(x=2.1972, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=-2.1972, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    # plt.legend()
    # plt.title(r'Calibrated models comparison -- Bayes error')
    # # plt.show()
    # plt.savefig('latex/images/calibrated_bayes_error_best_models_ev.pdf', format='pdf')



    ########################################################################################################################
    # Fusion

    print('\nFusion on validation set')
    score = np.vstack([scoresQuadLogReg, scoresRBFSVM, scoresGMMdiag])
    folds, foldlab, SCAL, LCAL = [], [], [], []
    for i in range(KFOLD): 
        folds.append(score[:, i::KFOLD])
        foldlab.append(LTE[i::KFOLD])

    for i in range(KFOLD):
        train = np.hstack([folds[j] for j in range(KFOLD) if j != i])
        trainlab = np.hstack([foldlab[j] for j in range(KFOLD) if j != i])
        LCAL.append(foldlab[i])

        w, b = trainWeightedLogRegBinary(train, trainlab, 0, pT)
        res = w.T @ folds[i] + b - np.log(pT / (1-pT))
        SCAL.append(res)
    
    
    SCAL = np.hstack(SCAL)
    LCAL = np.hstack(LCAL)
    res = (SCAL > 0) * 1
    acc = np.mean(res == LCAL)
    _, dcf = compute_bayes_risk_binary(SCAL, LCAL, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(SCAL, LCAL, pT, Cfn, Cfp)
    print(f"Accuracy: {acc * 100:.2f}% - DCF: {dcf:.3f} - minDCF: {mindcf:.3f}")

    # x, y, z = plot_Bayes_errorXXX(SCAL, LCAL, -4, 4, 100)
    # plt.plot(x, y, label='Fused actDCF', color='tab:blue')
    # plt.plot(x, z, label='Fused minDCF', color='tab:blue', linestyle='dashed')
    # plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    # plt.ylabel("DCF", fontsize=12)
    # plt.xlim(-4, 4)
    # plt.ylim(0, 1.19)
    # plt.axvline(x=2.1972, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=-2.1972, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    # plt.legend()
    # plt.title(r'Fused models comparison -- Bayes error')
    # # plt.show()
    # plt.savefig('latex/images/fusion_bayes_error_best_models_va.pdf', format='pdf')



    # Fusion on evaluation set
    scores = np.vstack([scoresQuadLogReg, scoresRBFSVM, scoresGMMdiag])
    scoresEval = np.vstack([scoresEvalQuadLogReg, scoresEvalRBFSVM, scoresEvalGMMdiag])

    print('\nFusion on evaluation set')
    w, b = trainWeightedLogRegBinary(scores, LTE, 0, pT)
    res = w.T @ scoresEval + b - np.log(pT / (1-pT))
    _, dcf = compute_bayes_risk_binary(res, LEVAL, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(res, LEVAL, pT, Cfn, Cfp)
    A = (res > 0) * 1
    acc = np.mean(A == LEVAL)
    print(f"Accuracy: {acc * 100:.2f}% - DCF: {dcf:.3f} - minDCF: {mindcf:.3f}")

    # x, y, z = plot_Bayes_errorXXX(res, LEVAL, -4, 4, 100)
    # plt.plot(x, y, label='Fused actDCF', color='tab:blue')
    # plt.plot(x, z, label='Fused minDCF', color='tab:blue', linestyle='dashed')
    # plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    # plt.ylabel("DCF", fontsize=12)
    # # plt.xlim(-4, 4)
    # # plt.ylim(0, 1.19)
    # plt.axvline(x=2.1972, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=-2.1972, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    # plt.legend()
    # plt.title(r'Fused models comparison -- Bayes error')
    # # plt.show()
    # plt.savefig('latex/images/fusion_bayes_error_best_models_ev.pdf', format='pdf')




########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    main(m_PCA=5, m_LDA=4, applyPCA=False, applyLDA=False, center=False)
