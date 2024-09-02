import numpy as np
from .logpdf_GAU_ND import loglikelihood
from .data_management import *


def binaryMVG(name, DTR: np.array, LTR: np.array, DTE: np.array, LTE: np.array, classes: list, pT=0.5, naive=False):
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
            
    logSJoint = logS
        
    llr = logSJoint[1] - logSJoint[0] - np.log(pT / (1 - pT))
    predictions = np.int32(llr > 0)

    accuracy = np.sum(predictions == LTE) / DTE.shape[1]
    print(f"{name}: {accuracy * 100:.2f}%")
    return logSJoint[1] - logSJoint[0]


def binaryMVGTied(name, DTR: np.array, LTR: np.array, DTE: np.array, LTE: np.array, classes: list, pT=0.5):
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
    
    logSJointTied = logSTied
    
    llr = logSJointTied[1] - logSJointTied[0] - np.log(pT / (1 - pT))
    predictionsMVG_Tied = np.int32(llr > 0)
    
    accuracyMVG_Tied = np.sum(predictionsMVG_Tied == LTE) / DTE.shape[1]
    print(f"{name}: {accuracyMVG_Tied * 100:.2f}%")

    return logSJointTied[1] - logSJointTied[0]


def binaryMVGnaive(name, DTR: np.array, LTR: np.array, DTE: np.array, LTE: np.array, classes: list, pT=0.5):
    return binaryMVG(name, DTR, LTR, DTE, LTE, classes, pT, naive=True)
