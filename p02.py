import numpy as np
import matplotlib.pyplot as plt
from src.mlpr_functions.data_management import *
from src.mlpr_functions.visualizer import *

def main():
    D, L = load_data('data_et_checks/trainData.txt')
    DEVAL, LEVAL = load_data('data_et_checks/evalData.txt')
    classes = ['Counterfeit', 'Genuine']
    
    DM = D - compute_mu_C(D)[0]
    visualize_pairwise(DM, L, np.array([0,1]), classes, a=.09)


if __name__ == '__main__': main()