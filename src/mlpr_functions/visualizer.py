from matplotlib import pyplot as plt
import numpy as np

def visualize_pairwise(matrix: np.array, label: np.array, classes: list, class_names: list, n=None, a=0.5, bins=50):
    if n is None: n = matrix.shape[0]
    
    if n == 0:
        print('matrix provided is zero dimensioned')
        return
    
    if n == 1:
        plt.figure()
        for cls in range(len(classes)):
             plt.hist(matrix[0, label==classes[cls]], density=1, bins=10, label=class_names[cls], alpha=0.65)
        # plt.savefig('img/LDA_1.pdf', format='pdf')
        return
    
    fig, axs = plt.subplots(n, n, figsize=(20,16))
    for i in range(n):
        for j in range(n):
            if i == j: # same feature extraction -> histogram
                for cls in range(len(classes)):
                    axs[i][j].hist(matrix[i, label==classes[cls]], density=1, bins=bins, label=class_names[cls], alpha=0.65)
                axs[i][j].set_title(f"Feature {i}", fontsize=8)
            else: # -> scatter plot
                for cls in range(len(classes)):
                    axs[i][j].scatter(matrix[i, label==classes[cls]], matrix[j, label==cls], label=class_names[cls], marker='.', alpha=a)
                axs[i][j].set_title(f"Feature {i} x Feature {j}", fontsize=8)

            axs[i][j].legend(fontsize=6)

    fig.tight_layout(pad=15/n)
    plt.show()
    # plt.savefig('latex/images/pairwise_visualization.pdf', format='pdf')