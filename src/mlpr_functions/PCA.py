import numpy as np
from .data_management import compute_mu_C

def PCA(matrix: np.array, dim: int):

    if dim > matrix.shape[0]:
        print("dimensions parameter provided is too large")
        return
    elif dim < 1:
        print("dimensions parameter provided is too small")
        return
    
    _, C = compute_mu_C(matrix)
    U, _, _ = np.linalg.svd(C)

    P = U[:, 0:dim]
    # x_PCA = P @ y_PCA # projection over initial (full) space

    return P, P.T @ matrix

def apply_pca(P, D):
    return P.T @ D