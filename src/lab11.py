import numpy as np
import matplotlib.pyplot as plt
from mlpr_functions.data_management import *
from mlpr_functions.GMM import *
from mlpr_functions.BayesRisk import *
from mlpr_functions.LogisticRegression import *


def main():

#-------------------------------------------------------------------#

    # data

    scores1 = np.load('../data_et_checks/data11/scores_1.npy')
    scores2 = np.load('../data_et_checks/data11/scores_2.npy')
    labels = np.load('../data_et_checks/data11/labels.npy')
    eval_scores1 = np.load('../data_et_checks/data11/eval_scores_1.npy')
    eval_scores2 = np.load('../data_et_checks/data11/eval_scores_2.npy')
    eval_labels = np.load('../data_et_checks/data11/eval_labels.npy')
    pT = 0.2


    # Bayes Risk analysis
    print('\nFull scores dataset')
    _, dcf = compute_bayes_risk_binary(scores1, labels, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(scores1, labels, 0.2, 1, 1)
    val = (scores1 > 0) * 1
    acc = np.mean(val == labels)
    print(f'System 1 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')
    
    _, dcf = compute_bayes_risk_binary(scores2, labels, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(scores2, labels, 0.2, 1, 1)
    val = (scores2 > 0) * 1
    acc = np.mean(val == labels)
    print(f'System 2 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')

    # Bayes error plot
    # x, y, z = plot_Bayes_errorXXX(scores1, labels, -3, 3, 50)
    # plt.plot(x, y, label='Scores 1 - actDCF', c='tab:blue')
    # plt.plot(x, z, label='Scores 1 - minDCF', c='tab:blue', linestyle='dashed')
 
    # x, y, z = plot_Bayes_errorXXX(scores2, labels, -3, 3, 50)
    # plt.plot(x, y, label='Scores 2 - actDCF', c='tab:orange')
    # plt.plot(x, z, label='Scores 2 - minDCF', c='tab:orange', linestyle='dashed')
 
    # plt.legend()
    # plt.xlim(-3.2, 3.2)
    # plt.ylim(0, 1.39)
    # plt.show()


#-------------------------------------------------------------------#

    # Calibration - single fold approach

    SCAL1, SVAL1 = scores1[::3], np.hstack((scores1[1::3], scores1[2::3]))
    SCAL2, SVAL2 = scores2[::3], np.hstack((scores2[1::3], scores2[2::3]))
    LCAL, LVAL = labels[::3], np.hstack((labels[1::3], labels[2::3]))

    # Bayes Risk analysis
    print('\nSplitted RAW dataset')
    _, dcf = compute_bayes_risk_binary(SVAL1, LVAL, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(SVAL1, LVAL, 0.2, 1, 1)
    val = (SVAL1 > 0) * 1
    acc = np.mean(val == LVAL)
    print(f'System 1 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')
    
    _, dcf = compute_bayes_risk_binary(scores2, labels, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(scores2, labels, 0.2, 1, 1)
    val = (scores2 > 0) * 1
    acc = np.mean(val == labels)
    print(f'System 2 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')

    # Calibration
    print('\nSplitted Calibrated dataset')
    w1, b1 = trainWeightedLogRegBinary(vrow(SCAL1), vrow(LCAL), 0, pT)
    RES1 = w1 @ vrow(SVAL1) + b1 - np.log(pT / (1 - pT))
    _, dcf = compute_bayes_risk_binary(RES1, LVAL, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(RES1, LVAL, 0.2, 1, 1)
    val = (RES1 > 0) * 1
    acc = np.mean(val == LVAL)
    print(f'System 1 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')

    pT = 0.2
    w2, b2 = trainWeightedLogRegBinary(vrow(SCAL2), vrow(LCAL), 0, pT)
    RES2 = w2 @ vrow(SVAL2) + b2 - np.log(pT / (1 - pT))
    _, dcf = compute_bayes_risk_binary(RES2, LVAL, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(RES2, LVAL, 0.2, 1, 1)
    val = (RES2 > 0) * 1
    acc = np.mean(val == LVAL)
    print(f'System 2 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')

    print('\nSplitted Calibrated dataset - using eval scores')
    REVAL1 = w1 @ vrow(eval_scores1) + b1 - np.log(pT / (1 - pT))
    _, dcf = compute_bayes_risk_binary(REVAL1, eval_labels, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(REVAL1, eval_labels, 0.2, 1, 1)
    val = (REVAL1 > 0) * 1
    acc = np.mean(val == eval_labels)
    print(f'System 1 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')
    # x, y, z = plot_Bayes_errorXXX(REVAL1, eval_labels, -3, 3, 50)
    # plt.plot(x, y, label='Scores 1 - actDCF', c='tab:blue')
    # plt.plot(x, z, label='Scores 1 - minDCF', c='tab:blue', linestyle='dashed')
    # x, y, _ = plot_Bayes_errorXXX(eval_scores1, eval_labels, -3, 3, 50)
    # plt.plot(x, y, label='Scores 1 - actDCF', c='tab:blue', linestyle='dotted')
    # plt.legend()
    # plt.xlim(-3.2, 3.2)
    # plt.ylim(0, .8)
    # plt.show()

    REVAL2 = w2 @ vrow(eval_scores2) + b2 - np.log(pT / (1 - pT))
    _, dcf = compute_bayes_risk_binary(REVAL2, eval_labels, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(REVAL2, eval_labels, 0.2, 1, 1)
    val = (REVAL2 > 0) * 1
    acc = np.mean(val == eval_labels)
    print(f'System 2 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')


#-------------------------------------------------------------------#

    # Calibration - k fold approach
    
    K = 5
    fold1, fold2 = [] , []
    foldlabels = []
    for i in range(K):
        fold1.append(scores1[i::K])
        fold2.append(scores2[i::K])
        foldlabels.append(labels[i::K])
    
    SCAL1, SCAL2, LCAL = [], [], []
    for i in range(K):
        lab = np.hstack([foldlabels[j] for j in range(K) if j != i])
        LCAL.append(foldlabels[i])

        train = np.hstack([fold1[j] for j in range(K) if j != i])
        evals = fold1[i]
        w, b = trainWeightedLogRegBinary(vrow(train), lab, 0, 0.2)
        res = w @ vrow(evals) + b - np.log(0.2 / 0.8)
        SCAL1.append(res)

        train = np.hstack([fold2[j] for j in range(K) if j != i])
        evals = fold2[i]
        w, b = trainWeightedLogRegBinary(vrow(train), lab, 0, 0.2)
        res = w @ vrow(evals) + b - np.log(0.2 / 0.8)
        SCAL2.append(res)
    
    SCAL1 = np.hstack(SCAL1)    # shape: (2000,)
    SCAL2 = np.hstack(SCAL2)    # shape: (2000,)
    LCAL = np.hstack(LCAL)      # shape: (2000,)

    print('\nK-fold Calibrated dataset')
    res1 = (SCAL1 > 0) * 1
    acc1 = np.mean(res1 == LCAL)
    _, dcf = compute_bayes_risk_binary(SCAL1, LCAL, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(SCAL1, LCAL, 0.2, 1, 1)
    print(f'System 1 -- accuracy: {acc1 * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')

    res2 = (SCAL2 > 0) * 1
    acc2 = np.mean(res2 == LCAL)
    _, dcf = compute_bayes_risk_binary(SCAL2, LCAL, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(SCAL2, LCAL, 0.2, 1, 1)
    print(f'System 2 -- accuracy: {acc2 * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')

    # retrain the models
    print('\nK-fold Calibrated dataset - using eval scores')
    w, b = trainWeightedLogRegBinary(vrow(scores1), labels, 0, 0.2)
    res = w @ vrow(eval_scores1) + b - np.log(0.2 / 0.8)
    # x, y, z = plot_Bayes_errorXXX(res, eval_labels, -3, 3, 150)
    # plt.plot(x, y, label='Scores 1 - actDCF', c='tab:blue')
    # plt.plot(x, z, label='Scores 1 - minDCF', c='tab:blue', linestyle='dashed')
    # x, y, _ = plot_Bayes_errorXXX(eval_scores1, eval_labels, -3, 3, 50)
    # plt.plot(x, y, label='Scores 1 - actDCF', c='tab:blue', linestyle='dotted')
    # plt.legend()
    # plt.xlim(-3.2, 3.2)
    # plt.ylim(0, .8)
    # plt.show()
    _, dcf = compute_bayes_risk_binary(res, eval_labels, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(res, eval_labels, 0.2, 1, 1)
    res = (res > 0) * 1
    acc = np.mean(res == eval_labels)
    print(f'System 1 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')
    

    w, b = trainWeightedLogRegBinary(vrow(scores2), labels, 0, 0.2)
    res = w @ vrow(eval_scores2) + b - np.log(0.2 / 0.8)
    _, dcf = compute_bayes_risk_binary(res, eval_labels, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(res, eval_labels, 0.2, 1, 1)
    res = (res > 0) * 1
    acc = np.mean(res == eval_labels)
    print(f'System 2 -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')

#-------------------------------------------------------------------#

    # Fusion
    scores = np.vstack((scores1, scores2))
    eval_scores = np.vstack((eval_scores1, eval_scores2))
    w, b = trainWeightedLogRegBinary(scores, labels, 0, pT)
    res = w @ eval_scores + b - np.log(pT / (1 - pT))
    
    # x, y, z = plot_Bayes_errorXXX(res, eval_labels, -3, 3, 150)
    # plt.plot(x, y, label='Fused - actDCF', c='tab:blue')
    # plt.plot(x, z, label='Fused - minDCF', c='tab:blue', linestyle='dashed')
    # 
    # x, y, z = plot_Bayes_errorXXX(eval_scores1, eval_labels, -3, 3, 50)
    # plt.plot(x, y, label='Scores 1 - actDCF', c='tab:orange')
    # plt.plot(x, z, label='Scores 1 - minDCF', c='tab:orange', linestyle='dashed')
    # 
    # x, y, z = plot_Bayes_errorXXX(eval_scores2, eval_labels, -3, 3, 50)
    # plt.plot(x, y, label='Scores 2 - actDCF', c='tab:green')
    # plt.plot(x, z, label='Scores 2 - minDCF', c='tab:green', linestyle='dashed')
    # 
    # plt.legend()
    # plt.xlim(-3.2, 3.2)
    # plt.ylim(0, .8)
    # plt.show()

    _, dcf = compute_bayes_risk_binary(res, eval_labels, 0.2, 1, 1)
    mindcf, _ = compute_minDCF_binary(res, eval_labels, 0.2, 1, 1)
    res = (res > 0) * 1
    acc = np.mean(res == eval_labels)
    print(f'Fusion -- accuracy: {acc * 100:.2f}, DCF: {dcf:.3f}, minDCF: {mindcf:.3f}')




#-------------------------------------------------------------------#



if __name__ == '__main__':
    main()