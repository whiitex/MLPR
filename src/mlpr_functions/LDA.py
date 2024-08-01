import numpy as np
import scipy.linalg
from .data_management import vcol, vrow


def LDA(matrix: np.array, L: np.array, classes, dimensions, orthogonal=False):

    if dimensions > matrix.shape[0]:
        print("dimensions parameter provided is too large")
        return
    elif dimensions < 1:
        print("dimensions parameter provided is too small")
        return
    
    SB, SW = compute_Sb_Sw(matrix, L)
    
    _, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:dimensions]

    if orthogonal:
        UW, _, _ = np.linalg.svd(W)
        W = UW[:, 0:dimensions]
    
    return W, W.T @ matrix

# S_b Separation BETWEEN classes 
# S_w Separation WITHIN classes
def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in np.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T

    # Sb, Sw
    return Sb / D.shape[1], Sw / D.shape[1]

def apply_lda(U, D):
    return U.T @ D