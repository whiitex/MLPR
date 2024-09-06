import numpy as np
from mlpr_functions.BayesRisk import *
from mlpr_functions.data_management import *


def main():

#-------------------------------------------------------------------#

    # Confusion Matrix

    ll = np.load('../data_et_checks/data7/commedia_ll.npy')
    labels = np.load('../data_et_checks/data7/commedia_labels.npy')
    classes = [0, 1, 2]

    pred = np.argmax(ll, axis=0)
    ConfMatrix = np.zeros((len(classes), len(classes)))


    for i in range(labels.shape[0]):
        ConfMatrix[pred[i], labels[i]] += 1

    # print(ConfMatrix)


#-------------------------------------------------------------------#

    # Binary optimal Bayes decision
    # DCFu and DCF - BINARY

    llr = np.load('../data_et_checks/data7/commedia_llr_infpar.npy')
    labels = np.load('../data_et_checks/data7/commedia_labels_infpar.npy')
    # 1 = inferno, 0 = paradiso

    pT, Cfn, Cfp = 0.5, 1, 1
    pred_lab = predict_optimal_Bayes_risk(llr, pT, Cfn, Cfp)
    cf = compute_confusion_matrix(pred_lab, labels)
    DCFu, DCF = compute_bayes_risk_binary(llr, labels, pT, Cfn, Cfp)
    # print(cf)
    print(f'{DCFu:.3f}, {DCF:.3f}')



#-------------------------------------------------------------------#

    # Compute min DCF

    llr = np.load('../data_et_checks/data7/commedia_llr_infpar.npy')
    labels = np.load('../data_et_checks/data7/commedia_labels_infpar.npy')
    
    # pT, Cfn, Cfp = 0.5, 1, 1
    DCFmin, thmin = compute_minDCF_binary(llr, labels, pT, Cfn, Cfp)
    print(f'DCFmin: {DCFmin:.3f}, thmin: {thmin:.3f}')
    print(f'Calibartion loss: {DCF - DCFmin:.3f}')
    
    
    
#-------------------------------------------------------------------#

    # ROC curve

    llr = np.load('../data_et_checks/data7/commedia_llr_infpar.npy')
    labels = np.load('../data_et_checks/data7/commedia_labels_infpar.npy')

    # plot_ROC_curve(llr, labels)



#-------------------------------------------------------------------#

    # Bayes error plotting

    llr = np.load('../data_et_checks/data7/commedia_llr_infpar.npy')
    labels = np.load('../data_et_checks/data7/commedia_labels_infpar.npy')

    plot_Bayes_error(llr, labels, -3, 3, 51)




if __name__ == '__main__':
    main()