import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Data from the table
PCA = [6, 5, 4, 3, 2, 1]
MVG_tied_actDCF = [0.400, 0.398, 0.460, 0.468, 0.443, 0.478]
MVG_tied_minDCF = [0.342, 0.351, 0.415, 0.439, 0.438, 0.434]
MVG_naive_actDCF = [0.463, 0.463, 0.462, 0.457, 0.479, 0.481]
MVG_naive_minDCF = [0.442, 0.445, 0.444, 0.434, 0.435, 0.434]
MVG_actDCF = [0.389, 0.466, 0.463, 0.459, 0.442, 0.478]
MVG_minDCF = [0.351, 0.434, 0.431, 0.434, 0.432, 0.434]

# Plotting the data with the same color for each MVG type
plt.figure(figsize=(7.5, 5))

# MVG
plt.plot(PCA, MVG_tied_actDCF, marker='o', linestyle='-', color='blue', label='MVG actDCF')
plt.plot(PCA, MVG_tied_minDCF, marker='o', linestyle='--', color='blue', label='MVG minDCF')

# MVG tied
plt.plot(PCA, MVG_naive_actDCF, marker='o', linestyle='-', color='green', label='MVG Tied actDCF')
plt.plot(PCA, MVG_naive_minDCF, marker='o', linestyle='--', color='green', label='MVG Tied minDCF')

# MVG naive
plt.plot(PCA, MVG_actDCF, marker='o', linestyle='-', color='red', label='MVG Naive actDCF')
plt.plot(PCA, MVG_minDCF, marker='o', linestyle='--', color='red', label='MVG Naive minDCF')

# Adding labels and title
plt.xlabel('PCA Components', fontsize=13)
plt.ylabel('DCF Values', fontsize=13)
plt.title('Comparison of actDCF and minDCF across Different MVG Types')
plt.legend()
plt.grid(True)
plt.xlim((0.9,6.1))
# plt.show()
plt.savefig('latex/images/MVG_bayesdecision_pi_09.pdf', format='pdf')