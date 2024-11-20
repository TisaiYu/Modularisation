import numpy as np
from PyQt5.QtCore import QThread,pyqtSignal
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform,pdist
from userInterface.ui.PlotCanvas import PlotCanvas

'''
算法部分
'''

class DrawDendrogram(QThread):
    draw_finished = pyqtSignal(np.ndarray, list,
                               PlotCanvas)  # TisaiYu[2024/6/20] 信号定义为类的属性，而不是实例属性，是因为连接要写在实例化调用函数之前，这样实例化的函数发出信号才可以被响应

    def __init__(self, DSM):
        super(DrawDendrogram, self).__init__()
        self.DSM = DSM
        self.dedrogram_view = PlotCanvas()
        self.dedrogram_view.axes.set_title('Dendrogram')
        self.dedrogram_view.axes.set_xlabel('Customers')
        self.dedrogram_view.axes.set_ylabel('distances')

    def run(self):
        dist_m1 = 1 - self.DSM
        # 将距离矩阵转换为压缩形式
        condensed_distance_matrix = squareform(dist_m1, checks=False)
        dis_mat = pdist(self.DSM,
                        'cosine')  # TisaiYu[2024/6/25] 层次聚类输入要么是一维数组（表示距离矩阵的压缩，比如30*30关联度矩阵，距离矩阵有30*30但是对称只取450），或者是二维数组（就是特征矩阵）

        # TisaiYu[2024/9/10] 因为功能对于很多零部件都有关联度的影响，而连接只对一些少数零部件有影响，而ward是类内方差链接法，因此对于功能这种对很多零部件有影响的大片区域数据，能够有所识别，因为功能影响大片，所以是方差稳定的主要因素，所以能先根据功能初始得到模块方案，再迭代求解
        Z = hierarchy.linkage(dis_mat, method='ward',
                              metric="euclidean")  # TisaiYu[2024/9/5] 当输入是一个一维的压缩距离矩阵时（使用pdist的结果），那么metric是没有用，如果是一个二维的表示输入特征的数组，metric实际和pdist做了一样的事情
        distances = Z[:, 2]
        reversed_last = distances[::-1]
        indexes = np.arange(1, len(distances) + 1)
        # 计算距离增长加速度,或者采用距离的最大变化
        accelerations = np.diff(reversed_last, 2)
        # accelerations = np.diff(reversed_last, 1)
        accelerations = np.pad(accelerations, (1, 1), 'constant', constant_values=(0, 0))  # 在首尾补零
        # 确定最佳聚类数
        optimal_cluster_num = np.argmax(accelerations) + 1  # +1 是因为 diff 函数减少了一个元素
        re = hierarchy.dendrogram(Z, color_threshold=Z[-(optimal_cluster_num-1),2], above_threshold_color='#bcbddc',
                                  ax=self.dedrogram_view.axes)
        num_array = [str(int(s)) for s in re["ivl"]]
        re["ivl"] = num_array
        input_sequence = re['ivl']
        print("输入零件的序号：", re["ivl"])
        self.dedrogram_view.axes.set_xticklabels(num_array)

        self.draw_finished.emit(Z, input_sequence, self.dedrogram_view)

