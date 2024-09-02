import scipy
import numpy as np
from .data_management import vcol, vrow


def train_dual_SVM_linear(DTR, LTR, C, K = 1):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = np.vstack([DTR, np.ones((1,DTR.shape[1])) * K])
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # USE THE ONE OF CUMANI
    def dual_func(alpha):
        J = 0.5 * (vrow(alpha) @ H @ vcol(alpha)) - alpha.sum()
        Grad = vcol(H @ alpha - 1)
        return J, Grad
    
    def primal_func(w):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * np.linalg.norm(w_hat)**2 + C * np.maximum(0, 1 - ZTR * S).sum()

    alphaStar, f, d = scipy.optimize.fmin_l_bfgs_b(func=dual_func, x0=np.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for _ in LTR], factr=1.0)

    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K

    dual_loss, primal_loss = -f, primal_func(w_hat)
    dual_gap = primal_loss - dual_loss
    results = [primal_loss, dual_loss, dual_gap]

    return w, b, results


def powerlogn(x, e: int):
    mul, ans = x, 1
    while e > 0:
        if e % 2 == 1:
            ans *= mul
        mul *= mul
        e >>= 1
    return ans

def polynomialKernel(degree: int, c):
    
    def polyKernelFunc(D1, D2):
        return powerlogn((np.dot(D1.T, D2) + c), degree)

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    return rbfKernelFunc

def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    kernelValue = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * kernelValue

    # USE THE ONE OF CUMANI
    def dual_func(alpha):
        J = 0.5 * (vrow(alpha) @ H @ vcol(alpha)) - alpha.sum()
        Grad = vcol(H @ alpha - 1)
        return J, Grad

    alphaStar, dualLoss, d = scipy.optimize.fmin_l_bfgs_b(func=dual_func, x0=np.zeros(DTR.shape[1]), bounds = [(0, C) for _ in LTR], factr=1.0)

    def fScore(DTE):
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore, dualLoss