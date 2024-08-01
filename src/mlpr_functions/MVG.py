import numpy as np
from .logpdf_GAU_ND import loglikelihood
from .data_management import vcol, vrow, compute_mu_C

def binaryMVG(name, DTR: np.array, LTR: np.array, DTE: np.array, LTE: np.array, classes: list, naive=False):
    mu = []
    sigma = []

    for i in range(len(classes)):
        Dc = DTR[:, LTR == i]
        muc, Cc = compute_mu_C(Dc)
        mu.append(muc)
        sigma.append(Cc)

    if naive:
        for c in range(len(classes)):
            sigma[c] *= np.eye(sigma[c].shape[c])
    

    logS = np.zeros((len(classes), DTE.shape[1]))
    for c in range(len(classes)):
        for i in range(DTE.shape[1]):
            logS[c,i] = loglikelihood(DTE[:, i].reshape(DTE.shape[0], 1), mu[c], sigma[c])
            
    # we are assuming that the classes are equiprobable, so P(c1) = P(c2) = 1/2, hence
    logSJoint = logS
    predictions = np.zeros((1, logSJoint.shape[1]))
    for i in range(DTE.shape[1]):
        if logSJoint[1,i] > logSJoint[0,i]:
            predictions[0,i] = 1
    
    accuracy = np.sum(predictions == LTE) / DTE.shape[1]
    print(f"{name}: {accuracy * 100:.2f}%")


def binaryMVGnaive(name, DTR: np.array, LTR: np.array, DTE: np.array, LTE: np.array, classes: list):
    binaryMVG(name, DTR, LTR, DTE, LTE, classes, naive=True)


def binaryMVGTied(name, DTR: np.array, LTR: np.array, DTE: np.array, LTE: np.array, classes: list):
    mu = []
    mu.append(DTR[:, LTR==0].mean(axis=1).reshape(DTR.shape[0], 1))
    mu.append(DTR[:, LTR==1].mean(axis=1).reshape(DTR.shape[0], 1))

    sigma = (DTR[:, LTR==0] - mu[0]) @ (DTR[:, LTR==0] - mu[0]).T 
    sigma += (DTR[:, LTR==1] - mu[1]) @ (DTR[:, LTR==1] - mu[1]).T
    sigma /= DTR.shape[1]

    logSTied = np.zeros((len(classes), DTE.shape[1]))
    for i in range(len(classes)):
        for j in range(DTE.shape[1]):
            logSTied[i,j] = loglikelihood(DTE[:, j].reshape(DTE.shape[0], 1), mu[i], sigma)
    
    # as before, we are assuming both classes to be equiprobable, so P(c1) = P(c2) = 1/2
    logSJointTied = logSTied
    predictionsMVG_Tied = np.zeros((1, logSJointTied.shape[1]))
    for i in range(DTE.shape[1]):
        if logSJointTied[1,i] > logSJointTied[0,i]:
            predictionsMVG_Tied[0,i] = 1
    
    accuracyMVG_Tied = np.sum(predictionsMVG_Tied == LTE) / DTE.shape[1]
    print(f"{name}: {accuracyMVG_Tied * 100:.2f}%")
