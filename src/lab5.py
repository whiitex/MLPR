import numpy as np
import scipy
from mlpr_functions.logpdf_GAU_ND import loglikelihood, likelihood, logpdf_GAU_ND
from mlpr_functions.data_management import split_db_2to1


def load_iris(path: str):
    file = open(path, 'r')
    matrix = []
    label = []
    for line in file:
        line = line.split(sep=',')
        matrix.append(line[0:4])
        name : int
        if line[4] == 'Iris-setosa\n': name = 0
        elif line[4] == 'Iris-versicolor\n': name = 1
        else: name = 2
        label.append(name)
    
    matrix = np.array(matrix, dtype='float32').T
    label = np.array(label, dtype=int)
    file.close()
    return (matrix, label)


def main():
    classes = [0,1,2]
    D, L = load_iris('src/iris.csv')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    mu = []
    Sigma = []

    for i in classes: # the 3 given classes [0. 1. 2]
        D_lab = DTR[:, LTR == i]
        mu.append(D_lab.mean(axis=1).reshape(D.shape[0], 1))
        Sigma.append(1 / D_lab.shape[1] * (D_lab - mu[i]) @ (D_lab - mu[i]).T)
        # print(mu[i])
        # print(Sigma[i])

    #########################################################################
    # likelihood (not log)

    # S[i, j] is the depend probability of class 'i' given sample 'j' 
    S = np.zeros((len(classes), DTE.shape[1]))
    for c in classes:
        for j in range(DTE.shape[1]):
            S[c,j] = likelihood(DTE[:, j].reshape(DTE.shape[0], 1), mu[c], Sigma[c])
        
    SJoint = S / 3

    # check
    # check = np.load('data_et_checks/data5/SJoint_MVG.npy')
    # print(np.allclose(check, SJoint))


    SMarginal = SJoint.sum(axis=0).reshape((1, SJoint.shape[1]))
    SPost = SJoint / SMarginal

    Pred = np.argmax(SPost, axis=0)
    correct_num = error_num = 0
    for i in range(len(Pred)):
        if Pred[i] == LTE[i]:
            correct_num += 1
        else: error_num += 1

    # print(f"Correct rate: {correct_num / len(Pred) * 100}%")
    # print(f"Error rate: {error_num / len(LTE) * 100}%")


    #########################################################################
    # LOG-likelihood 

    logS = np.zeros((len(classes), DTE.shape[1]))
    for c in classes:
        for j in range(DTE.shape[1]):
            logS[c,j] = loglikelihood(DTE[:, j].reshape(DTE.shape[0], 1), mu[c], Sigma[c])

    logSJoint = logS + np.log(1/3)

    logSMarginal = np.zeros((1, logSJoint.shape[1]))
    # SMarginal_log2 = np.zeros((1, SJoint_log.shape[1]))
    for i in range(logSMarginal.shape[1]):
        maxl = np.argmax(logSJoint[:, i])
        tobelogged = 0
        for c in range(len(classes)):
            tobelogged += np.exp(logSJoint[c, i] - maxl)
        logSMarginal[0, i] = maxl + np.log(tobelogged)
        # SMarginal_log2[0,i] = scipy.special.logsumexp(SJoint_log[:, i])

    logSPost = logSJoint - logSMarginal
    logPred = np.argmax(logSPost, axis=0)

    logcorrect_num = logerror_num = 0
    for i in range(len(Pred)):
        if logPred[i] == LTE[i]:
            logcorrect_num += 1
        else: logerror_num += 1
    
    # print(f"Correct rate: {logcorrect_num / len(logPred) * 100}%")
    # print(f"Error rate: {logerror_num / len(LTE) * 100}%")



    #########################################################################
    # Naive Bayes

    Sigma_naive = []
    mu_naive = []

    for i in classes:
        mu_naive.append(DTR[:, LTR == i].mean(axis=1).reshape(D.shape[0], 1))
        s_sq = DTR[:, LTR == i].var(axis=1)
        Sigma_naive.append(s_sq * np.eye(D.shape[0]))

    Sigma_naive[0] = np.identity(4) * Sigma[0]
    Sigma_naive[1] = np.identity(4) * Sigma[1]
    Sigma_naive[2] = np.identity(4) * Sigma[2]
    logS_naive = np.zeros((len(classes), DTE.shape[1]))
    for c in classes:
        for j in range(DTE.shape[1]):
            logS_naive[c,j] = loglikelihood(DTE[:, j].reshape(DTE.shape[0], 1), mu_naive[c], Sigma_naive[c])


    logSJoint_naive = logS_naive + np.log(3)


    logSMarginal_naive = np.zeros((1, logSJoint_naive.shape[1]))
    for i in range(logSMarginal.shape[1]):
        logSMarginal_naive[0,i] = scipy.special.logsumexp(logSJoint_naive[:, i])

    logSPost_naive = logSJoint_naive - logSMarginal_naive

    # check = np.load('data_et_checks/data5/logPosterior_NaiveBayes.npy')
    # print(np.allclose(check, logSPost_naive))

    logPred_naive = np.argmax(logSPost_naive, axis=0)

    logcorrect_num_naive = 0
    for i in range(len(Pred)):
        if logPred_naive[i] == LTE[i]:
            logcorrect_num_naive += 1
    logerror_num_naive = len(LTE) - logcorrect_num_naive
    # print(f"Correct rate: {logcorrect_num_naive / len(logPred_naive) * 100}%")
    # print(f"Error rate: {logerror_num_naive / len(LTE) * 100}%")



    #########################################################################
    # Tied - Common Covariance

    Sigma_common = np.zeros((4,4))

    for j in range(DTR.shape[1]):
        mult = DTR[:, j].reshape(DTR.shape[0], 1) - mu[LTR[j]]
        Sigma_common += mult @ mult.T
    Sigma_common /= DTR.shape[1]

    logS_common = np.zeros((len(classes), DTE.shape[1]))
    for i in classes:
        for j in range(DTE.shape[1]):
            logS_common[i,j] = loglikelihood(DTE[:, j].reshape(DTE.shape[0], 1), mu[i], Sigma_common)
    
    logSJoint_common = logS_common + np.log(1/3)

    logSMarginal_common = np.zeros((1, logSJoint_common.shape[1]))
    for i in range(logSMarginal_common.shape[1]):
        logSMarginal_common[0,i] = scipy.special.logsumexp(logSJoint_common[:, i])

    logSPost_common = logSJoint_common - logSMarginal_common

    logPred_common = np.argmax(logSPost_common, axis=0)
    logcorrect_num_common = logerror_num_common = 0
    for i in range(DTE.shape[1]):
        if logPred_common[i] == LTE[i]:
            logcorrect_num_common += 1
        else: logerror_num_common += 1

    # print(f"Correct rate: {logcorrect_num_common / len(logPred_naive) * 100}%") # 98.0%
    # print(f"Error rate: {logerror_num_common / len(LTE) * 100}%") # 2.0%



if __name__ == '__main__':
    main()