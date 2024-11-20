import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# 示例数据
data = np.random.rand(10, 2)  # 10个样本，2个特征
Z = linkage(data, method='ward')  # 进行层次聚类

# 假设每个样本的类别标签
classified_labels = ['A', 'A', 'B', 'B', 'C', 'A', 'C', 'C', 'B', 'A                                                                                                                                                                                                                                             ']

# 创建颜色映射
unique_labels = set(classified_labels)
color_map = {label: plt.cm.jet(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

# 绘制 dendrogram
plt.figure(figsize=(10, 6))
dendro = dendrogram(Z, color_threshold=0)  # 先绘制不带颜色的 dendrogram

# 为每个叶子节点分配颜色
for i, d in enumerate(dendro['dcoord']):
    # 获取当前节点的标签
    label_index = int(dendro['ivl'][i])  # 根据索引获取标签
    label = classified_labels[label_index]  # 获取标签
    # 绘制线条并设置颜色
    plt.plot(dendro['icoord'][i], d, color=color_map[label])

# 设置聚类节点以上的颜色为灰色
for i, d in enumerate(dendro['dcoord']):
    height = d[1]  # 当前节点的高度
    # 获取当前节点的标签
    label_index = int(dendro['ivl'][i])  # 根据索引获取标签
    label = classified_labels[label_index]  # 获取标签

    # 检查是否是聚类节点
    if i < len(dendro['dcoord']) - 1:  # 确保不是最后一个节点
        next_height = dendro['dcoord'][i + 1][1]  # 下一个节点的高度
        if height < next_height:  # 如果当前节点的高度小于下一个节点
            plt.plot([dendro['icoord'][i][0], dendro['icoord'][i][1]], [height, height], color='gray', linestyle='--')

plt.title('Dendrogram with Custom Colors')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()