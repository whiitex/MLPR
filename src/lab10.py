import numpy as np
import matplotlib.pyplot as plt
from mlpr_functions.data_management import *
from mlpr_functions.GMM import *
from mlpr_functions.BayesRisk import *
from mlpr_functions.BayesRisk import *
import json


def load_iris(path: str):
    file = open(path, 'r')
    matrix = []
    label = []
    for line in file:
        line = line.split(sep=',')
        matrix.append(line[0:4])
        name : int
        if line[4] == 'Iris-setosa\n': name = 0
        elif line[4] == 'Iris-versicolor\n': name = 1
        else: name = 2
        label.append(name)
    
    matrix = np.array(matrix, dtype='float32').T
    label = np.array(label, dtype=int)
    file.close()
    return (matrix, label)


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


def main():



#-------------------------------------------------------------------#

    # data
    data4d = np.load('../data_et_checks/data10/GMM_data_4D.npy')
    data1d = np.load('../data_et_checks/data10/GMM_data_1D.npy')
    gmm_init4d = load_gmm('../data_et_checks/data10/GMM_4D_3G_init.json')
    gmm_init1d = load_gmm('../data_et_checks/data10/GMM_1D_3G_init.json')


#-------------------------------------------------------------------#

    # GMM - EM algorithm
    
    # GMM 4D
    check = np.load('../data_et_checks/data10/GMM_4D_3G_init_ll.npy')
    # logpdf = logpdf_GMM(data4d, gmm_init4d)
    # print(np.allclose(logpdf, check))

    check = load_gmm('../data_et_checks/data10/GMM_4D_3G_EM.json')
    gmm = train_GMM_EM(data4d, gmm_init4d, verbose=False)
    # for i in range(len(gmm)):
        # for g in range(3):
            # print(np.allclose(gmm[i][g], check[i][g]))
    print(f'average ll 4d: {logpdf_GMM(data4d, gmm).mean()}')

    
    # GMM 1D
    # plt.hist(vcol(data1d).ravel(), 40, alpha=0.5, edgecolor='black', density=True)
    # 
    gmm = train_GMM_EM(data1d, gmm_init1d, verbose=False)
    y = logpdf_GMM(np.sort(data1d, axis=1), gmm)
    print(f'average ll 1d: {y.mean()}')
    # plt.plot(np.sort(data1d, axis=1).ravel(), np.exp(y), linewidth=2)
    # plt.xlim(-10, 5)
    # plt.show()


#-------------------------------------------------------------------#

    # GMM - LBG algorithm

    # GMM 4D
    gmm = train_GMM_LBG_EM(data4d, 4, verbose=False)
    check = load_gmm('../data_et_checks/data10/GMM_4D_4G_EM_LBG.json')
    # for i in range(len(gmm)):
        # for g in range(3):
            # print(np.allclose(gmm[i][g], check[i][g]))
    print(f'average ll 4d lsb: {logpdf_GMM(data4d, gmm).mean()}')


    # GMM 1D
    # plt.hist(vcol(data1d).ravel(), 50, edgecolor='black', density=True)
    gmm1dlsb = train_GMM_LBG_EM(data1d, 64, verbose=False)
    y = logpdf_GMM(np.sort(data1d, axis=1), gmm1dlsb)
    print(f'average ll 1d lsb: {y.mean()}')
    # plt.plot(np.sort(data1d, axis=1).ravel(), np.exp(y), linewidth=2)
    # plt.show()


#-------------------------------------------------------------------#

    # Iris dataset
    classes = [0,1,2]
    D, L = load_iris('iris.csv')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


#-------------------------------------------------------------------#

    # GMM - binary task
    databinary = np.load('../data_et_checks/data10/ext_data_binary.npy')
    labelbinary = np.load('../data_et_checks/data10/ext_data_binary_labels.npy')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(databinary, labelbinary)

    clusters = 8
    func = 'full'
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], clusters, covType=func, verbose=False)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], clusters, covType=func, verbose=False)
    llr = classiy_GMM_binary(DTE, gmm0, gmm1)
    LVAL = (llr > 0) * 1
    acc = np.mean(LVAL == LTE)
    print(f'accuracy: {acc * 100:.2f}%')

    _, dcf = compute_bayes_risk_binary(llr, LTE, 0.5, 1.0, 1.0)
    mindcf, _ = compute_minDCF_binary(llr, LTE, 0.5, 1.0, 1.0)
    print(f'dcf: {dcf:.4f}, mindcf: {mindcf:.4f}')



#-------------------------------------------------------------------#



if __name__ == '__main__':
    main()