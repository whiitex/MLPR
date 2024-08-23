import scipy
import numpy as np
from mlpr_functions.data_management import *

def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0
    
    def obj_func(v):
        w, b = v[0:-1], v[-1]
        s = (vcol(w).T @ DTR).ravel() + b

        J = np.logaddexp(0, -ZTR * s)
        R = l / 2 * np.linalg.norm(w) ** 2 + J.mean()

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        Gw = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = np.mean(G)
        Grad = np.hstack([Gw, Gb])

        return R, Grad


    x, f, _ = scipy.optimize.fmin_l_bfgs_b(obj_func, x0=np.zeros(DTR.shape[0] + 1), approx_grad=False, maxfun=10000000, factr=1000.0)
    
    print(f"Log-reg - lambda = {l:.5e} - J*(w, b) = {f:.5e}")
    
    w_opt, b_opt = x[0:-1], x[-1]
    return w_opt, b_opt


def trainWeightedLogRegBinary(DTR, LTR, l, prior):

    ZTR = LTR * 2.0 - 1.0
    wTrue = prior / (ZTR>0).sum() 
    wFalse = (1-prior) / (ZTR<0).sum()

    def obj_func(v):
        w, b = v[0:-1], v[-1]
        s = (vcol(w).T @ DTR).ravel() + b

        J = np.logaddexp(0, -ZTR * s)
        J[ZTR > 0] *= wTrue
        J[ZTR < 0] *= wFalse

        R = l / 2 * np.linalg.norm(w) ** 2 + J.sum()

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTrue
        G[ZTR < 0] *= wFalse

        Gw = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        Grad = np.hstack([Gw, Gb])

        return R, Grad
    
    x, f, _ = scipy.optimize.fmin_l_bfgs_b(obj_func, x0=np.zeros(DTR.shape[0] + 1), approx_grad=False, maxfun=10000000, factr=1000.0)

    print(f"Weighted Log-reg (pT {prior:.5e}) - lambda = {l:.5e} - J*(w, b) = {f:.5e}")

    w_opt, b_opt = x[0:-1], x[-1]
    return w_opt, b_opt