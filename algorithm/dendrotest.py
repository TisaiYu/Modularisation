import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# 示例数据
data = np.random.rand(50, 4)  # 50个输入
indx = [33,0,25,43,22,40]
subset_data = data[indx]  # 取其中的一个子集（如前10个输入）

# 对整个数据集进行聚类
Z_full = sch.linkage(data, method='single')

# 对子集进行聚类
Z_subset = sch.linkage(subset_data, method='single')

# 绘制聚类树进行比较
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sch.dendrogram(Z_full, ax=ax1)
ax1.set_title('Full Data Set Dendrogram')
sch.dendrogram(Z_subset, ax=ax2,labels=indx)
ax2.set_title('Subset Data Set Dendrogram')
plt.show()
