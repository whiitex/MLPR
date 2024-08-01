import numpy as np

def logpdf_GAU_ND(X: np.array, mu: np.array, C: np.array):
    M = X.shape[0]
    _, C_det = np.linalg.slogdet(C)
    C_inv = np.linalg.inv(C)

    const = -M/2 * np.log(2 * np.pi)
    const -= C_det / 2
    
    # ans = np.zeros(X.shape[1])
    # for i in range(X.shape[1]):
    #     row = X[:,i].reshape(M,1)
    #     logdensity = -(row - mu).T @ C_inv @ (row - mu)/2 + const
        
    #     ans[i] = logdensity

    # return ans
    return -M/2 * np.log(np.pi*2) - 0.5*C_det - 0.5 * ((X-mu) * (C_inv @ (X-mu))).sum(0)


def likelihood(X, m, C):
    GAU_vector = np.exp(logpdf_GAU_ND(X, m, C)) 
    return GAU_vector.sum()


def loglikelihood(X, m, C):
    logGAU_vector = logpdf_GAU_ND(X, m, C) 
    return logGAU_vector.sum()
