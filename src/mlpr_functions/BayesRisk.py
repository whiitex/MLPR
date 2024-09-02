import numpy as np
import matplotlib.pyplot as plt

def predict_labels_binary(llr, threshold):
    return np.int32(llr > threshold)

# Predict labels according to Bayes minimum risk
def predict_optimal_Bayes_risk(llr, pT, Cfn, Cfp):
    t = -np.log(pT * Cfn / (1 - pT) / Cfp)
    return predict_labels_binary(llr, t)

# Compute confusion matrix
def compute_confusion_matrix(predictedLabels, labels, nClasses=2):
    M = np.zeros((nClasses, nClasses))
    for i in range(len(labels)):
        M[int(predictedLabels[i]), int(labels[i])] += 1
    return M

# Compute both DCFu and DCF
def compute_bayes_risk_binary(llr, labels, prior, Cfn, Cfp):
    predictedLabels = predict_optimal_Bayes_risk(llr, prior, Cfn, Cfp)
    M = compute_confusion_matrix(predictedLabels, labels, 2)
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[1,0] + M[0,0])

    DCFu = Pfn * prior * Cfn + Pfp * (1 - prior) * Cfp
    DCF = DCFu / min(prior * Cfn, (1 - prior) * Cfp)

    return DCFu, DCF

# Compute min DCF and the corresponding threshold
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter] 
    classLabelsSorted = classLabels[llrSorter] 

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1: nFalseNegative += 1 
        if classLabelsSorted[idx] == 0: nFalsePositive -= 1 
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: 
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut) # we return also the corresponding thresholds
    
def compute_minDCF_binary(llr, classLabels, prior, Cfn, Cfp):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1-prior)*Cfp) 

    idx = np.argmin(minDCF)
    return minDCF[idx], th[idx]


def plot_ROC_curve(llr, labels):
    xfp, ytp = [], []
    for i in np.concatenate([np.array([-np.inf]), llr, np.array([np.inf])]):
        pred_lab = predict_labels_binary(llr, i)
        cf = compute_confusion_matrix(pred_lab, labels)
        Ptp = cf[1,1] / (cf[1,1] + cf[0,1])
        Pfp = cf[1,0] / (cf[1,0] + cf[0,0])
        xfp.append(Pfp)
        ytp.append(Ptp)

    plt.plot(sorted(xfp), sorted(ytp), linewidth=2, color='blue')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.show()

def plot_Bayes_error(llr, labels, lrange, rrange, npoints = 30):
        xplot, yplot, yplotmin = [], [], []
        for i in np.linspace(lrange, rrange, npoints):
            effP = 1 / (1 + np.exp(-i))
            pred_lab = predict_labels_binary(llr, effP)
            _, dcf = compute_bayes_risk_binary(pred_lab, labels, effP, 1, 1)
            dcfmin, _ = compute_minDCF_binary(llr, labels, effP, 1, 1)
            xplot.append(i)
            yplot.append(dcf)
            yplotmin.append(dcfmin)
        
        plt.plot(xplot, yplot, linewidth=2, color='blue', label='DCF')
        plt.plot(xplot, yplotmin, linewidth=2, color='red', label='min DCF')
        plt.xlim(lrange, rrange)
        plt.ylim(0, (rrange - lrange) / 2.5)
        plt.legend()
        plt.show()

def plot_Bayes_errorXXX(llr, labels, lrange, rrange, npoints = 30):
        xplot, yplot, yplotmin = [], [], []
        for i in np.linspace(lrange, rrange, npoints):
            effP = 1 / (1 + np.exp(-i))
            pred_lab = predict_labels_binary(llr, effP)
            _, dcf = compute_bayes_risk_binary(pred_lab, labels, effP, 1, 1)
            dcfmin, _ = compute_minDCF_binary(llr, labels, effP, 1, 1)
            xplot.append(i)
            yplot.append(dcf)
            yplotmin.append(dcfmin)
        
        return xplot, yplot, yplotmin