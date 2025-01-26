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
    # Score loading

    scoresQuadLogReg = np.load('models/llrQuadraticLogReg.npy')
    scoresPoly4SVM = np.load('models/llrPoly4SVM.npy')
    scoresRBFSVM = np.load('models/llrRBFSVM.npy')
    scoresGMMdiag = np.load('models/llrGMMdiagonal.npy')
    
    scoresEvalQuadLogReg = np.load('models/llrEvalQuadraticLogReg.npy')
    scoresEvalPoly4SVM = np.load('models/llrEvalPoly4SVM.npy')
    scoresEvalRBFSVM = np.load('models/llrEvalRBFSVM.npy')
    scoresEvalGMMdiag = np.load('models/llrEvalGMMdiagonal.npy')

    #########################################################################
    # K-fold cross validation on GMM diag

    pT, Cfn, Cfp = 0.1, 1, 1

    KFOLD = 10

    # K-fold on validation set
    print("K-fold on validation set")
    colors = ['tab:orange', 'tab:green', 'tab:blue']
    funcs = ['SVMpoly4', 'SVMRBF', 'GMMdiag']
    a = 0
    for score in [scoresPoly4SVM, scoresRBFSVM, scoresGMMdiag]:
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

        x, y, z = plot_Bayes_errorXXX(SCAL, LCAL, -3.8, 3.8, 100)
        plt.plot(x, y, label=f'{funcs[a]} cal actDCF', color=colors[a])
        plt.plot(x, z, label=f'{funcs[a]} minDCF', color=colors[a], linestyle='dashed')
        x, y, _ = plot_Bayes_errorXXX(folds[i], foldlab[i], -3.8, 3.8, 100)
        plt.plot(x, y, label=f'{funcs[a]} raw actDCF', color=colors[a], linestyle='dotted')
        a += 1
    
    plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    plt.ylabel("DCF", fontsize=12)
    # plt.xlim(-4, 4)
    # plt.ylim(0, 1.19)
    plt.axvline(x=2.1972, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=-2.1972, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.title(r'Calibrated models comparison -- Bayes error')
    plt.show()
    # plt.savefig('latex/images/calibrated_bayes_error_best_models_va.pdf', format='pdf')

    
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
    w, b = trainWeightedLogRegBinary(vrow(scoresGMMdiag), LTE, 0, pT)
    res = w.T @ vrow(scoresEvalGMMdiag) + b - np.log(pT / (1-pT))
    x, y, z = plot_Bayes_errorXXX(res, LEVAL, -4, 4, 100)
    plt.plot(x, y, label='GMMdiag cal actDCF', color='tab:blue')
    plt.plot(x, z, label='GMMdiag minDCF', color='tab:blue', linestyle='dashed')
    x, y, _ = plot_Bayes_errorXXX(scoresEvalGMMdiag, LEVAL, -4, 4, 100)
    plt.plot(x, y, label='GMMdiag raw actDCF', color='tab:blue', linestyle='dotted')
    
    w, b = trainWeightedLogRegBinary(vrow(scoresPoly4SVM), LTE, 0, pT)
    res = w.T @ vrow(scoresEvalPoly4SVM) + b - np.log(pT / (1-pT))
    x, y, z = plot_Bayes_errorXXX(res, LEVAL, -4, 4, 100)
    plt.plot(x, y, label='SVMpoly4 cal actDCF', color='tab:orange')
    plt.plot(x, z, label='SVMpoly4 minDCF', color='tab:orange', linestyle='dashed')
    x, y, _ = plot_Bayes_errorXXX(scoresEvalPoly4SVM, LEVAL, -4, 4, 100)
    plt.plot(x, y, label='SVMpoly4 raw actDCF', color='tab:orange', linestyle='dotted')

    w, b = trainWeightedLogRegBinary(vrow(scoresRBFSVM), LTE, 0, pT)
    res = w.T @ vrow(scoresEvalRBFSVM) + b - np.log(pT / (1-pT))
    x, y, z = plot_Bayes_errorXXX(res, LEVAL, -4, 4, 100)
    plt.plot(x, y, label='RBFSVM cal actDCF', color='tab:green')
    plt.plot(x, z, label='RBFSVM minDCF', color='tab:green', linestyle='dashed')
    x, y, _ = plot_Bayes_errorXXX(scoresEvalRBFSVM, LEVAL, -4, 4, 100)
    plt.plot(x, y, label='RBFSVM raw actDCF', color='tab:green', linestyle='dotted')

    plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    plt.ylabel("DCF", fontsize=12)
    # plt.xlim(-4, 4)
    # plt.ylim(0, 1.19)
    plt.axvline(x=2.1972, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=-2.1972, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.title(r'Calibrated models comparison -- Bayes error')
    plt.show()
    # plt.savefig('latex/images/calibrated_bayes_error_best_models_ev.pdf', format='pdf')



    #########################################################################
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

    x, y, z = plot_Bayes_errorXXX(SCAL, LCAL, -4, 4, 100)
    plt.plot(x, y, label='Fused actDCF', color='tab:blue')
    plt.plot(x, z, label='Fused minDCF', color='tab:blue', linestyle='dashed')
    plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    plt.ylabel("DCF", fontsize=12)
    # plt.xlim(-4, 4)
    # plt.ylim(0, 1.19)
    plt.axvline(x=2.1972, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=-2.1972, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.title(r'Fused models comparison -- Bayes error')
    plt.show()
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

    x, y, z = plot_Bayes_errorXXX(res, LEVAL, -4, 4, 100)
    plt.plot(x, y, label='Fused actDCF', color='tab:blue')
    plt.plot(x, z, label='Fused minDCF', color='tab:blue', linestyle='dashed')
    plt.xlabel(r"$\log \frac{\tilde{\pi}}{1+-\tilde{\pi}}$", fontsize=12)
    plt.ylabel("DCF", fontsize=12)
    # plt.xlim(-4, 4)
    # plt.ylim(0, 1.19)
    plt.axvline(x=2.1972, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=-2.1972, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.title(r'Fused models comparison -- Bayes error')
    plt.show()
    # plt.savefig('latex/images/fusion_bayes_error_best_models_ev.pdf', format='pdf')
    


if __name__ == '__main__':
    main(m_PCA=5, m_LDA=4, applyPCA=False, applyLDA=False, center=False)
