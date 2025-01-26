import numpy as np
import matplotlib.pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *
from src.mlpr_functions.PCA import *
from src.mlpr_functions.LDA import *
from src.mlpr_functions.BayesRisk import *
from src.mlpr_functions.GMM import *


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
    # GMM

    pT, Cfn, Cfp = 0.1, 1, 1
    for func in ['full', 'diagonal', 'tied']:
        for cl1 in [1,2,4,8,16,32]:
            for cl2 in [1,2,4,8,16,32]:
                try:
                    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], cl1, covType=func, verbose=False)
                    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], cl2, covType=func, verbose=False)
                    llr = classiy_GMM_binary(DTE, gmm0, gmm1, pT)
                    SVAL = (llr > 0) * 1
                    acc, err = np.mean(SVAL == LTE), 1 - np.mean(SVAL == LTE)
                    _, actdcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
                    mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
                    print(f'comp: [{cl1}, {cl2}] \t- func: {func[:4]} - acc: {acc * 100:.2f}% - dcf: {actdcf:.3f}, mindcf: {mindcf:.3f}')
                except: 
                    print(f'error comp: [{cl1}, {cl2}]')
                    continue


    # GMM diagonal
    cl1, cl2, func = 4, 16, 'diagonal'
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], cl1, covType=func, verbose=False)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], cl2, covType=func, verbose=False)
    llr = classiy_GMM_binary(DTE, gmm0, gmm1, pT)
    SVAL = (llr > 0) * 1
    acc, err = np.mean(SVAL == LTE), 1 - np.mean(SVAL == LTE)
    _, actdcf = compute_bayes_risk_binary(llr, LTE, pT, Cfn, Cfp)
    mindcf, _ = compute_minDCF_binary(llr, LTE, pT, Cfn, Cfp)
    print(f'comp: [{cl1}, {cl2}] \t- func: {func[:4]} - acc: {acc * 100:.2f}% - dcf: {actdcf:.3f}, mindcf: {mindcf:.3f}')
    
    # llrGMMdiagonal = llr
    # llrEvalGMMdiagonal = classiy_GMM_binary(DEVAL, gmm0, gmm1, pT)
    # np.save('models/llrGMMdiagonal.npy', llrGMMdiagonal)
    # np.save('models/llrEvalGMMdiagonal.npy', llrEvalGMMdiagonal)


    #########################################################################
    # Model comparison on different application

    scoresQuadLogReg = np.load('models/llrQuadraticLogReg.npy')
    scoresPoly4SVM = np.load('models/llrPoly4SVM.npy')
    scoresRBFSVM = np.load('models/llrRBFSVM.npy')
    scoresGMMdiag = np.load('models/llrGMMdiagonal.npy')

    x,y,z = plot_Bayes_errorXXX(scoresQuadLogReg, LTE, -4, 4, 50)
    plt.plot(x, y, label='LogReg2 DCF', color='r')
    plt.plot(x, z, label='LogReg2 minDCF', color='r', linestyle='dashed')

    x,y,z = plot_Bayes_errorXXX(scoresPoly4SVM, LTE, -4, 4, 50)
    plt.plot(x, y, label='PolySVM4 DCF', color='b')
    plt.plot(x, z, label='PolySVM4 minDCF', color='b', linestyle='dashed')

    x,y,z = plot_Bayes_errorXXX(scoresRBFSVM, LTE, -4, 4, 50)
    plt.plot(x, y, label='RBFSVM DCF', color='m')
    plt.plot(x, z, label='RBFSVM minDCF', color='m', linestyle='dashed')

    x,y,z = plot_Bayes_errorXXX(scoresGMMdiag, LTE, -4, 4, 50)
    plt.plot(x, y, label='GMMdiag DCF', color='g')
    plt.plot(x, z, label='GMMdiag minDCF', color='g', linestyle='dashed')

    plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    plt.ylabel("DCF", fontsize=12)
    plt.xlim(-4, 4)
    plt.ylim(0, 1.19)
    plt.axvline(x=2.1972, color='black', linestyle='--')
    plt.axvline(x=-2.1972, color='black', linestyle='--')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.legend()
    plt.title(r'Best models comparison -- Bayes error')
    plt.show()
    # plt.savefig('latex/images/best_model_bayes_error.pdf', format='pdf')



if __name__ == '__main__':
    main(m_PCA=5, m_LDA=4, applyPCA=False, applyLDA=False, center=False)
