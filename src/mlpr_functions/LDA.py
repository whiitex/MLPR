import numpy as np
import scipy.linalg


def LDA(matrix: np.array, L: np.array, classes, dimensions, orthogonal=False):

    if dimensions > matrix.shape[0]:
        print("dimensions parameter provided is too large")
        return
    elif dimensions < 1:
        print("dimensions parameter provided is too small")
        return
    
    SB = __SB(matrix, L, classes)
    SW = __SW(matrix, L, classes)
    
    _, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:dimensions]

    if orthogonal:
        UW, _, _ = np.linalg.svd(W)
        W = UW[:, 0:dimensions]
    
    return W.T @ matrix


# S_b Separation BETWEEN classes 
def __SB(matrix, L, classes):

    mu = matrix.mean(axis=1).reshape(matrix.shape[0], 1)
    
    SB = np.zeros((matrix.shape[0], matrix.shape[0]))
    
    for i in classes:
        Dc = matrix[:, L == i]
        muc = Dc.mean(axis=1).reshape(Dc.shape[0], 1)
        SB = SB + ((muc - mu) @ (muc - mu).T) * Dc.shape[1]
    SB = SB / matrix.shape[1]
    return SB


# S_w Separation WITHIN classes
def __SW(matrix, L, classes):
    SW = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in classes:
        Dc = matrix[:, L == i]
        muc = Dc.mean(axis=1).reshape(Dc.shape[0], 1)
        Cc = (Dc - muc) @ (Dc - muc).T
        SW += Cc
    SW /= matrix.shape[1]
    return SW