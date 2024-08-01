from matplotlib import pyplot as plt
import numpy as np
from mlpr_functions.logpdf_GAU_ND import logpdf_GAU_ND, loglikelihood
from mlpr_functions.data_management import vcol, vrow


def main():
    
    #####################################################
    # Plot of the gaussian multivariate model

    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    # plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape(1,1000), m, C).ravel()))
    # plt.show()

    # check
    pdfSol = np.load('data_et_checks/data4/llGAU.npy')
    pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
    print(np.allclose(pdfSol, pdfGau))

    XND = np.load('data_et_checks/data4/XND.npy')
    mu = np.load('data_et_checks/data4/muND.npy')
    C = np.load('data_et_checks/data4/CND.npy')
    pdfSol = np.load('data_et_checks/data4/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(pdfGau.shape)
    print(np.abs(pdfSol - pdfGau).max())

    
    
    #####################################################
    # likelihood of a lognormal distribution
    
    # data: np.array = np.load('data_et_checks/data4/XND.npy')
    # a = np.array([1.2])
    # m_ML = data.mean(axis=1).reshape(data.shape[0], 1)
    # c_ML = ((data - m_ML) @ (data - m_ML).T) /data.shape[1]    
    # print(loglikelihood(data, m_ML, c_ML))

    
    
    #####################################################
    # mono-dimension data representation and likelihood
    
    data = np.load('data_et_checks/data4/X1D.npy')
    plt.figure()
    plt.hist(data.ravel(), bins=50, density=1, alpha=0.8)
    XPlot = np.linspace(-8, 12, 1000).reshape(1,1000)
    m_ML = data.mean(axis=1)
    c_ML = (data - m_ML) @ (data - m_ML).T / data.shape[1]
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot, m_ML, c_ML)))
    print(loglikelihood(data, m_ML, c_ML))
    plt.show()    


if __name__ == '__main__':
    main()