import numpy as np
from scipy.cluster.hierarchy import fcluster,dendrogram
from scipy.spatial.distance import squareform,pdist
import matplotlib.pyplot as plt
import pandas as pd
dsm = np.array(pd.read_excel(r"E:\Postgraduate\YY\code\11.10ModularizationPy\DSM.xlsx",header=None,index_col=None))

dis_mat = pdist(dsm,metric='cosine')
dist_matrix = squareform(dis_mat)
index_i, index_j = 126, 170 # 例如查询第1个样本和第3个样本之间的距离
distance_ij = dist_matrix[index_i, index_j]
print(f"样本{index_i}和样本{index_j}之间的距离: {distance_ij}")
index_i, index_j = 241, 105 # 例如查询第1个样本和第3个样本之间的距离
distance_ij = dist_matrix[index_i, index_j]
print(f"样本{index_i}和样本{index_j}之间的距离: {distance_ij}")
index_i, index_j = 241, 240 # 例如查询第1个样本和第3个样本之间的距离
distance_ij = dist_matrix[index_i, index_j]
print(f"样本{index_i}和样本{index_j}之间的距离: {distance_ij}")

distance_matrix =1-dsm
index_i, index_j = 165, 99 # 例如查询第1个样本和第3个样本之间的距离
distance_ij = distance_matrix[index_i, index_j]
print(f"样本{index_i}和样本{index_j}之间的距离: {distance_ij}")
index_i, index_j = 50, 99 # 例如查询第1个样本和第3个样本之间的距离
distance_ij = distance_matrix[index_i, index_j]
print(f"样本{index_i}和样本{index_j}之间的距离: {distance_ij}")

n = dsm.shape[0]
distance_array = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
if index_i > index_j:
    index_i, index_j = index_j, index_i
idx = n * index_i - (index_i + 1) * index_i // 2 + index_j - index_i - 1
print(f"压缩矩阵中的距离 ({index_i}, {index_j}): {distance_array[idx]}")