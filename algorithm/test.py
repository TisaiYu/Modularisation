import numpy as np
from scipy.cluster.hierarchy import fcluster,dendrogram
import matplotlib.pyplot as plt
aa = [[ 0. , 1.  ,1. , 2.], [ 1. ,2. , 1. , 3.], [ 4., 5. , 1.  ,4.]]
Z = np.array(aa)
re = dendrogram(Z,color_threshold=Z[0,2]-0.01)
plt.show()
