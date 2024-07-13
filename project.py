import numpy as np
from matplotlib import pyplot as plt
from src.mlpr_functions.data_management import split_db_2to1, load_data
from src.mlpr_functions.visualizer import visualize_pairwise
from src.mlpr_functions.PCA import PCA
from src.mlpr_functions.LDA import LDA
from src.mlpr_functions.logpdf_GAU_ND import logpdf_GAU_ND, loglikelihood


def main():
    D, L = load_data('./trainData.txt')
    classes = ['Counterfeit', 'Genuine']
    
    mu = D.mean(axis=1).reshape(D.shape[0], 1)
    DM = D - mu

    # Pair-wise scatter plots dependencies
    # visualize_pairwise(DM, L, np.array([0,1]))


    ############################################################
    # PCA - Principal Component Analysis

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)


    m_PCA = 6 # 4/5 should be good
    print(f"m_PCA: {m_PCA}")
    P_PCA, DTR_PCA = PCA(DTR, m_PCA)
    DTE_PCA = P_PCA.T @ DTE

    # visualize_pairwise(DTR_PCA, LTR, [0,1], classes, a=0.05, bins=40)



    ########################################################################################################################
    # LDA - Linear Discriminant Analysis
    

    m_LDA = 3
    print(f"m_LDA: {m_LDA}")
    W, DTR_LDA = LDA(DTR_PCA, LTR, [0,1], min(m_PCA, m_LDA), True)
    DTE_LDA = W.T @ DTE_PCA
    # visualize_pairwise(y_LDA, L, [0,1], classes, a=0.05) 

    mu0 = W[:, 0].T @ DTR_PCA[:, LTR == 0].mean(axis=1).reshape(DTR_PCA.shape[0], 1)
    mu1 = W[:, 0].T @ DTR_PCA[:, LTR == 1].mean(axis=1).reshape(DTR_PCA.shape[0], 1)

    predictionsLDA = np.zeros((DTE_PCA.shape[1]))
    for i in range(DTE.shape[1]):
        x = DTE_PCA[:, i].reshape(DTE_PCA.shape[0], 1)
        xT = W[:, 0].T @ x
        if np.linalg.norm(xT - mu0) > np.linalg.norm(xT - mu1):
            predictionsLDA[i] = 1

    accuracyLDA = np.sum(predictionsLDA == LTE) / DTE_PCA.shape[1]
    print(f"accuracyLDA: {accuracyLDA * 100}%")


    '''
    Using only the LDA model, we can see that the accuracy is 90.65%.
    This is a good result, however, if we use the PCA model before the LDA model,
    we can get up to 91.0% by using only 1 dimension in the PCA (LDA can only use
    1 dimensione because of the number of classes, so this meta-parameter cannot
    be tested). Number of features is limited, but, as we can evaluate by results,
    the direction having the highest grade of variance is the most important one.

    LDA -> 90.65%
    PCA(1) + LDA  ->  91.00%
    PCA(2) + LDA  ->  90.95%
    PCA(3) + LDA  ->  90.55%
    PCA(4) + LDA  ->  90.70%
    PCA(5) + LDA  ->  90.55%
    PCA(6) + LDA  ->  90.55%
    '''


    ########################################################################################################################
    # Gaussian Multivariate density fitting

    fig, axs = plt.subplots(4, 3, figsize=(20,16))
    for c in range(2):
        Dc = np.sort(DTR[:, LTR == c], axis=1)
        mu = Dc.mean(axis=1).reshape(Dc.shape[0], 1)
        for i in range(DTR.shape[0]):
            row = Dc[i,:].reshape(1, Dc.shape[1])
            Sigma = (row - mu[i]) @ (row - mu[i]).T / row.shape[1]
            Sigma = np.ones((1,1)) * Sigma
            axs[c*2 + i//3][i%3].hist(row.ravel(), label=classes[c], density=1, bins=50, alpha=.8)
            axs[c*2 + i//3][i%3].plot(row.ravel(), np.exp(logpdf_GAU_ND(row, mu[i], Sigma)), linewidth=1.75)
            axs[c*2 + i//3][i%3].set_title(f"Feature {i+1}", fontsize=10)
            axs[c*2 + i//3][i%3].legend(fontsize=8)

    fig.tight_layout(pad=5)
    # plt.show()

    '''
    As we can see, the Gaussian Multivariate density fitting is working well with
    the given data, in particular it fits pretty good feauture 1 to 4. Instead, 
    features 5 and 6 seems to be the most discriminant, because they act very differently
    with respect with a MVG reference. For this reason, we can expect a quite good result
    from the bayesian classifier.
    '''


    ########################################################################################################################
    # MVG model classification

    mu = []
    Sigma = []

    # Since there a lot of samples and only three features, we can use
    # the vanilla MVG, so no Naive or Tied to build the model
    for i in range(len(classes)):
        D_lab = DTR_PCA[:, LTR == i]
        mu.append(D_lab.mean(axis=1).reshape(DTR_PCA.shape[0], 1))
        Sigma.append(1 / D_lab.shape[1] * (D_lab - mu[i]) @ (D_lab - mu[i]).T)
    
    logS = np.zeros((len(classes), DTE_PCA.shape[1]))
    for c in range(len(classes)):
        for i in range(DTE_PCA.shape[1]):
            logS[c,i] = loglikelihood(DTE_PCA[:, i].reshape(DTE_PCA.shape[0], 1), mu[c], Sigma[c])
            
    # we are assuming that the classes are equiprobable, so P(c1) = P(c2) = 1/2, hence
    logSJoint = logS
    predictionsMVG = np.zeros((1, logSJoint.shape[1]))
    for i in range(DTE_PCA.shape[1]):
        if logSJoint[1,i] > logSJoint[0,i]:
            predictionsMVG[0,i] = 1
    
    accuracyMVG = np.sum(predictionsMVG == LTE) / DTE_PCA.shape[1]
    print(f"accuracyMVG: {accuracyMVG * 100}%")
    

    '''
    MVG classifier performs very well, in particular it achieved a 2% more accuracy
    with respect to the LDA model. This is due to the fact that dara fits well this
    distribution, thus, as expected, we got an improvement.
    In this case, using PCA to remove non discriminant features is not useful and, 
    as we can see, the accuracy is slowly decreasing with the number of features. However,
    the result with respecting with LDA is always better, meaning that the MVG model is 
    better performing than the LDA model. Even using an LDA (pre-cleaned with PCA) gets
    worse results

    MVG -> 93.00%
    PCA(6) + MVG  ->  92.90%
    PCA(5) + MVG  ->  92.85%
    PCA(4) + MVG  ->  92.00%
    PCA(3) + MVG  ->  90.95%
    PCA(2) + MVG  ->  91.05%
    PCA(1) + MVG  ->  91.05%
    '''
    

    ########################################################################################################################
    # MVG TIED model classification

    muTied = []
    muTied.append(DTR_PCA[:, LTR==0].mean(axis=1).reshape(DTR_PCA.shape[0], 1))
    muTied.append(DTR_PCA[:, LTR==1].mean(axis=1).reshape(DTR_PCA.shape[0], 1))

    SigmaTied = (DTR_PCA[:, LTR==0] - muTied[0]) @ (DTR_PCA[:, LTR==0] - muTied[0]).T 
    SigmaTied += (DTR_PCA[:, LTR==1] - muTied[1]) @ (DTR_PCA[:, LTR==1] - muTied[1]).T
    SigmaTied /= DTR_PCA.shape[1]

    logSTied = np.zeros((len(classes), DTE_PCA.shape[1]))
    for i in range(len(classes)):
        for j in range(DTE_PCA.shape[1]):
            logSTied[i,j] = loglikelihood(DTE_PCA[:, j].reshape(DTE_PCA.shape[0], 1), muTied[i], SigmaTied)
    
    # as before, we are assuming both classes to be equiprobable, so P(c1) = P(c2) = 1/2
    logSJointTied = logSTied
    predictionsMVG_Tied = np.zeros((1, logSJointTied.shape[1]))
    for i in range(DTE_PCA.shape[1]):
        if logSJointTied[1,i] > logSJointTied[0,i]:
            predictionsMVG_Tied[0,i] = 1
    
    accuracyMVG_Tied = np.sum(predictionsMVG_Tied == LTE) / DTE_PCA.shape[1]
    print(f"accuracyMVG_Tied: {accuracyMVG_Tied * 100}%")
    
    '''
    The MVG Tied model has lower performance with respect to the MVG model and, as expected,
    it is quite perfectly following the LDA model. This is due to the fact that the tied model
    is actually built using the Within-Class Covariance matrix, which is the same as the LDA model.

    MVG Tied -> 90.7%
    PCA(1) + MVG Tied  ->  91.00%
    PCA(2) + MVG Tied  ->  90.90%
    PCA(3) + MVG Tied  ->  90.65%
    PCA(4) + MVG Tied  ->  90.70%
    PCA(5) + MVG Tied  ->  90.55%
    PCA(6) + MVG Tied  ->  90.70%
    '''



if __name__ == '__main__':
    main()