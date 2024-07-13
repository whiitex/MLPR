import numpy as np

def PCA(matrix: np.array, dimensions: int):

    if dimensions > matrix.shape[0]:
        print("dimensions parameter provided is too large")
        return
    elif dimensions < 1:
        print("dimensions parameter provided is too small")
        return
    
    mu = matrix.mean(axis=1).reshape(matrix.shape[0], 1)
    DM = matrix - mu

    C = DM @ DM.T / DM.shape[1]
    U, _, _ = np.linalg.svd(C)

    P = U[:, 0:dimensions]
    y_PCA = P.T @ DM
    # x_PCA = P @ y_PCA # projection over initial (full) space

    return y_PCA