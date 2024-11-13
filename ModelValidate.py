import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


plt.rcParams['font.family'] = ['Microsoft YaHei']  # 设置字体为微软雅黑


def elbow_curve():


    # 生成示例数据
    np.random.seed(42)
    data = np.random.rand(50, 2)

    # 生成层次聚类的链接矩阵 Z
    Z = linkage(data, method='ward')

    # 计算每个聚类数下的总距离
    last = Z[-10:, 2]
    reversed_last = last[::-1]
    indexes = np.arange(1, len(last) + 1)

    # 计算距离增长加速度
    accelerations = np.diff(reversed_last, 2)
    accelerations = np.pad(accelerations, (1, 1), 'constant', constant_values=(0, 0))  # 在首尾补零

    # 绘制 Elbow 曲线和加速度变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(indexes, reversed_last, marker='o', label='距离')
    plt.plot(indexes, accelerations, marker='x', label='定义的加速度')
    plt.xlabel('模块数量')
    plt.ylabel('值')
    plt.title('模块设计的最佳分割点确定')
    plt.legend()
    plt.show()

    # 确定最佳聚类数
    optimal_cluster_num = np.argmax(accelerations) + 1  # +1 是因为 diff 函数减少了一个元素
    print(f'最佳聚类数: {optimal_cluster_num}')


if __name__ == "__main__":
    elbow_curve()