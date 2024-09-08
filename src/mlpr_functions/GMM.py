import scipy
import numpy as np
from .data_management import vcol, vrow, compute_mu_C


def logpdf_GAU_ND(x, mu, C): # Fast version from Lab 4
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)


# gmm = [w, mu, cov]
def logpdf_GMM(X, gmm):
    S = []
    for w, mu, s in gmm:
        logSjoint = np.log(w) + logpdf_GAU_ND(X, mu, s)
        S.append(logSjoint)
    
    return scipy.special.logsumexp(S, axis=0)


def smooth_covariance_matrix(C, psi):

    U, s, Vh = np.linalg.svd(C)
    s[s<psi]=psi
    return U @ (vcol(s) * U.T)


def train_GMM_EM_Iteration(X, gmm, covType = 'Full', psiEig = None): 

    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    
    # E-step
    S = []
    for w, mu, C in gmm:
        logSjoint = logpdf_GAU_ND(X, mu, C) + np.log(w)
        S.append(logSjoint)
        
    S = np.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)

    resps = np.exp(S - logdens)

    # M-step
    gmm_new = []
    for idx in range(len(gmm)): 

        gamma = resps[idx]
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))
        S = (vrow(gamma) * X) @ X.T
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal': # Diagonal covariance matrix
            CUpd  = CUpd * np.eye(X.shape[0]) 
        gmm_new.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied': # Tied covariance matrix
        CTied = 0
        for w, mu, C in gmm_new:
            CTied += w * C
        gmm_new = [(w, mu, CTied) for w, mu, C in gmm_new]

    if psiEig is not None:
        gmm_new= [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmm_new]
        
    return gmm_new


def train_GMM_EM(X, gmm, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, verbose=True):

    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if verbose:
        print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType = covType, psiEig = psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

    if verbose:
        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))        
    return gmm
    

def split_GMM_LBG(gmm, alpha = 0.1, verbose=True):

    gmmOut = []
    if verbose:
        print ('LBG - going from %d to %d components' % (len(gmm), len(gmm)*2))
    for (w, mu, C) in gmm:
        U, s, _ = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut


def train_GMM_LBG_EM(X, numComponents, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, lbgAlpha = 0.1, verbose=True):

    mu, C = compute_mu_C(X)

    if covType.lower() == 'diagonal':
        C = C * np.eye(X.shape[0])
    
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))]
    else:
        gmm = [(1.0, mu, C)]
    
    while len(gmm) < numComponents:
        
        if verbose: print ('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        
        # Split the components
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        if verbose:
            print ('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm).mean())

        gmm = train_GMM_EM(X, gmm, covType = covType, psiEig = psiEig, verbose=verbose, epsLLAverage = epsLLAverage)
    return gmm


# def classify_GMM(X, gmms):
#     S = []
#     for w, mu, C in gmm:
#         logSjoint = logpdf_GAU_ND(X, mu, C) + np.log(w)
#         S.append(logSjoint)
    
#     return np.argmax(S, axis=0)


def classiy_GMM_binary(X, gmm0, gmm1, pT=0.5):
    S0 = logpdf_GMM(X, gmm0) + np.log(pT)
    S1 = logpdf_GMM(X, gmm1) + np.log(1-pT)

    return S1 - S0 # llr