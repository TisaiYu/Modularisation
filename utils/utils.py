import numpy as np
import pandas as pd
from scipy.cluster import hierarchy  # 导入层次聚类算法
import matplotlib
import time
import networkx as nx
from PyQt5 import QtSql
import itertools
from sklearn import metrics
from algorithm.FuzzyAssessMT import *
import matplotlib.pylab as plt
from collections import defaultdict

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# 生成7个矩阵s1到s7，这里只是示例，实际情况需要根据具体数据生成
def generate_symmetric_matrix1(size):
    # 生成一个大小为size*size的零矩阵
    matrix = np.zeros((size, size))

    # 生成对角线上全为1的矩阵
    np.fill_diagonal(matrix, 1)

    # 随机选择的数值列表
    choices = [0.8, 0.5, 0.2, 0]

    # 填充矩阵的上三角部分（不包括对角线）
    for i in range(size):
        for j in range(i + 1, size):
            # 随机选择一个数值
            value = np.random.choice(choices)
            # 填充对称位置
            matrix[i][j] = matrix[j][i] = value

    return matrix

def generate_symmetric_matrix2(size):
    # 生成一个大小为size*size的零矩阵
    matrix = np.zeros((size, size))

    # 生成对角线上全为1的矩阵
    np.fill_diagonal(matrix, 1)

    # 随机选择的数值列表
    choices = [0.8, 0.4, 0]

    # 填充矩阵的上三角部分（不包括对角线）
    for i in range(size):
        for j in range(i + 1, size):
            # 随机选择一个数值
            value = np.random.choice(choices)
            # 填充对称位置
            matrix[i][j] = matrix[j][i] = value

    return matrix

def generate_symmetric_matrix3(size):
    # 生成一个大小为size*size的零矩阵
    matrix = np.zeros((size, size))

    # 生成对角线上全为1的矩阵
    np.fill_diagonal(matrix, 1)

    # 随机选择的数值列表
    choices = [0.8, 0.3, 0]

    # 填充矩阵的上三角部分（不包括对角线）
    for i in range(size):
        for j in range(i + 1, size):
            # 随机选择一个数值
            value = np.random.choice(choices)
            # 填充对称位置
            matrix[i][j] = matrix[j][i] = value

    return matrix

def generate_test_value(n, num_classes, num_matrices=7):
    if n < num_classes:
        num_classes = n
    points_per_class = n // num_classes  # 每个类别的数据点数
    matrices = np.zeros((num_matrices,n, n))

    # 为每个类别生成高相关性数据
    for i in range(num_classes):
        start_index = i * points_per_class
        end_index = start_index + points_per_class if i < num_classes - 1 else n

        for matrix in matrices:
            # 类内高相关性
            high_correlation = np.random.uniform(0.8, 1.0, (end_index - start_index, end_index - start_index))
            matrix[start_index:end_index, start_index:end_index] = high_correlation

            # 确保对角线是1
            np.fill_diagonal(matrix[start_index:end_index, start_index:end_index], 1)

    # 为不同类别间生成低相关性数据
    for matrix in matrices:
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                si = i * points_per_class
                ei = si + points_per_class if i < num_classes - 1 else n
                sj = j * points_per_class
                ej = sj + points_per_class if j < num_classes - 1 else n

                # 类间低相关性
                low_correlation = np.random.uniform(0.0, 0.2, (ei - si, ej - sj))
                matrix[si:ei, sj:ej] = low_correlation
                matrix[sj:ej, si:ei] = low_correlation.T  # 对称填充

    return matrices


def shuffle_data(data):
    n = data.shape[0]
    original_indices = np.arange(n)
    np.random.shuffle(original_indices)
    shuffled_data = data[original_indices]
    return shuffled_data, original_indices


def compute_D1(cluster_labels,
               DSM):  # TisaiYu[2024/6/3] 模块内聚合度，越大越好，就是每个模块内的物品对应的距离矩阵上的加起来再除分母，分母为每个模块物品数量的任取2个组合，然后各个模块的取组合结果加起来。
    unique_labels = np.unique(cluster_labels)  # TisaiYu[2024/6/3] np.unique得到有几个类

    D1 = 0
    for label in unique_labels:
        D1_numerator = 0
        D1_denominator = 0
        indices = np.where(cluster_labels == label)[0]  # TisaiYu[2024/6/3] 结果是([i,j,k...],)所以要取[0]
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i+1,len(indices)):
                D1_numerator += DSM[indices[i], indices[j]]
        D1_denominator += ((len(indices) * (len(indices)-1))/2)
        D1 += (D1_numerator / D1_denominator)
    return D1/len(unique_labels)


def compute_D2(cluster_labels,
               DSM):  # TisaiYu[2024/6/3] 模块间耦合度，越小越好，同D1理，就是模块间的物品对应关联矩阵的值加起来，然后再除一个分母，分母是两两模块，各自的物品数量乘积，最后各个模块都两两组合，加起来。
    unique_labels = np.unique(cluster_labels)

    D2 = 0
    class_nums = len(unique_labels)
    for i in range(class_nums):
        D2_iner = 0
        for j in range(i + 1, class_nums):
            indices_i = np.where(cluster_labels == unique_labels[i])[0]
            indices_j = np.where(cluster_labels == unique_labels[j])[0]
            D2_numerator = 0
            D2_denominator = 0
            for idx_i in indices_i:
                for idx_j in indices_j:
                    D2_numerator += DSM[idx_i, idx_j]
            D2_denominator += (len(indices_i) * len(indices_j))
            D2_iner += (D2_numerator / D2_denominator)
        D2 += D2_iner
    if class_nums>1:
        return 2*D2/(class_nums*class_nums-1)
    else:
        return np.inf

def compute_CH(cluster_labels,DSM): # TisaiYu[2024/6/25] 关于聚类的数据指标，有些调库的输入是特征矩阵DSM，或者是距离矩阵，要辨别。
    # distance_matrix = 1 - DSM
    if len(np.unique(cluster_labels)) == 1:
        return 0
    # n_samples = len(cluster_labels)
    # n_clusters = len(np.unique(cluster_labels))
    # overall_mean = np.mean(distance_matrix)
    #
    # # 计算簇间方差 Bk，簇内方差Wk
    # Wk = 0
    # Bk = 0
    # for i in np.unique(cluster_labels):
    #     cluster_mask = (cluster_labels == i)
    #     nq = np.count_nonzero(cluster_mask)
    #     within_cluster_distances = distance_matrix[np.ix_(cluster_mask, cluster_mask)] # TisaiYu[2024/6/25] 这里重复了ij ji
    #     cq = np.mean(within_cluster_distances)
    #     Wk += np.sum((within_cluster_distances/2 - cq) ** 2)
    #     Bk += nq * (cq - overall_mean) ** 2
    #
    # # 计算CH指数
    # CH = ((n_samples - n_clusters) * Bk) / ((n_clusters - 1) * Wk)
    # x_mds = MDS(DSM)
    dia_value = np.max(DSM)
    # CH = metrics.calinski_harabasz_score(x_mds,cluster_labels)
    CH = metrics.calinski_harabasz_score(DSM,cluster_labels)
    return CH

def compute_DB(cluster_labels,DSM):
    if len(np.unique(cluster_labels)) == 1:
        return 0
    x_mds = MDS(DSM)
    dia_value = np.max(DSM)
    DB = metrics.davies_bouldin_score(x_mds,cluster_labels)
    # DB = metrics.davies_bouldin_score(DSM,cluster_labels)
    return DB


def compute_Sil(cluster_labels,DSM): # TisaiYu[2024/6/25] Sil是轮廓系数，但是其输入是原特征矩阵，不是距离矩阵
    if len(np.unique(cluster_labels)) == 1:
        return 0
    dia_value = np.max(DSM)
    Sil = metrics.silhouette_score(dia_value-DSM, cluster_labels,metric="precomputed")
    Sil = (Sil+1)/2 # TisaiYu[2024/6/28] 缩放到0到1
    return Sil

def compute_Gap(cluster_labels,DSM, B=20):
    dk = compute_Wk(DSM, cluster_labels)
    if dk==0:
        W_k = 0
    else:
        W_k = np.log(dk)
    nums = len(np.unique(cluster_labels))
    # 对随机数据进行聚类
    random_W_k = np.zeros(B)
    for i in range(B): # TisaiYu[2024/6/5] 平均20次来计算期望
        random_data = np.random.rand(*DSM.shape)
        Z = hierarchy.linkage(random_data, method='weighted', metric='sqeuclidean')
        random_data_labels = hierarchy.fcluster(Z, nums, criterion='maxclust') # TisaiYu[2024/6/5] Gap要求对随机采样的新数据执行和源数据相同的聚类方法
        random_W_k[i] = np.log(compute_Wk(random_data, random_data_labels))

    # 计算期望的对数簇间散度
    E_log_W_k = np.mean(random_W_k)

    # 计算Gap Statistic
    Gap_k = E_log_W_k - W_k # TisaiYu[2024/6/5] 随机采样的数据的聚类期望-源数据的散度，如果数值越大，说明聚类效果越好。越小则说明源数据接近均匀分布（因为随机采样是均匀分布），则没什么可分类的，说明很差

    return Gap_k

def compute_Wk(X, labels):
    W_k = 0
    for label in np.unique(labels):
        cluster = X[labels == label]
        centroid = np.mean(cluster, axis=0)
        W_k += np.sum((cluster - centroid)**2)
    return W_k

def compute_G(cluster_labels,DSM):
    unique_labels = np.unique(cluster_labels)
    G = 0
    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0]
        if len(indices) < 2:
            continue
        sum_distances = 0
        sum_ones = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sum_distances += DSM[indices[i], indices[j]]
                sum_ones += 1
        sum_all_ones = sum_ones * len(unique_labels)
        Gn = (sum_distances ** 2) / (sum_ones * sum_all_ones)
        G += Gn

    return G

def compute_CE(cluster_labels,DSM):
    unique_labels = np.unique(cluster_labels)
    s_out = 0
    s_in = 0
    s_in_count =0
    s_out_count=0
    for label in unique_labels: # TisaiYu[2024/6/13] 遍历模块内
        indices = np.where(cluster_labels == label)[0]
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dist = DSM[indices[i], indices[j]]
                if dist==0:
                    s_in += dist
                    s_in_count+=1
                else:
                    continue


    for i in range(len(unique_labels)): # TisaiYu[2024/6/13] 遍历模块间
        for j in range(i + 1, len(unique_labels)):
            indices_i = np.where(cluster_labels == unique_labels[i])[0]
            indices_j = np.where(cluster_labels == unique_labels[j])[0]
            for idx_i in indices_i:
                for idx_j in indices_j:
                    dist = DSM[idx_i, idx_j]
                    if dist!=0:
                        s_out += dist
                        s_out_count+=1
                    else:
                        continue
    s_in = s_in/s_in_count
    s_out = s_out/s_out_count
    if s_in==0 and s_out==0:
        CE = 1
    else:
        CE = 1/(0.5*s_in+0.5*s_out)
    return CE

def compute_Q(cluster_labels,DSM):
    G = nx.from_numpy_array(DSM)
    communities = []
    for label in set(cluster_labels):
        community = [i for i, x in enumerate(cluster_labels) if x == label]
        communities.append(set(community))
    Q = nx.community.modularity(G,communities)
    # unique_labels = np.unique(cluster_labels)
    # H = np.zeros((len(unique_labels),len(unique_labels)))
    # DSM_conjunc = np.copy(DSM)
    # DSM_conjunc[np.where(DSM >0.5)] = 1
    # all_sum = np.sum(DSM)
    # Q=0
    # for index,label in enumerate(unique_labels):  # TisaiYu[2024/6/13] 遍历模块内
    #     yii = 0
    #     indices = np.where(cluster_labels == label)[0]
    #     if len(indices) < 2:
    #         continue
    #     for i in range(len(indices)):
    #         for j in range(i + 1, len(indices)):
    #             yii += DSM[indices[i], indices[j]]
    #     yii = 2*yii/(len(indices)*(len(indices)-1))
    #     H[index,index] = yii/all_sum
    # for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块间
    #     for j in range(i + 1, len(unique_labels)):
    #         yij = 0
    #         indices_i = np.where(cluster_labels == unique_labels[i])[0]
    #         indices_j = np.where(cluster_labels == unique_labels[j])[0]
    #         for idx_i in indices_i:
    #             for idx_j in indices_j:
    #                 yij += DSM[idx_i, idx_j]
    #         H[i, j] = yij/all_sum/2
    #         H[j, i] = yij/all_sum/2
    # for i in range(len(unique_labels)):
    #     Q = Q+H[i,i]-np.sum(H[:,i])**2
    # print(Q)
    Q = (Q+1)/2
    return Q

def reencode_labels(labels):
    unique_labels = np.unique(labels)
    new_labels = np.zeros_like(labels)
    for i, label in enumerate(unique_labels):
        new_labels[labels == label] = i + 1
    return new_labels

def time_count(func,labels,matrix,print_step_time=False):
    if print_step_time:
        start_time = time.time()
        result = func(labels,matrix)
        end_time = time.time()
        print(f"{func.__name__}运行时间：{end_time-start_time}秒")
        return result
    else:
        return func(labels,matrix)


def MDS(DSM):
    from sklearn.manifold import MDS
    np.random.seed(0)   # 5个样本，每个样本10个特征
    from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity

    mds = MDS(n_components=2)
    X_mds = mds.fit_transform(DSM)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_mds[:, 0], X_mds[:, 1], s=100, color='steelblue', alpha=0.8)
    # plt.title('2D MDS Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.grid(True, linestyle='dotted')
    # plt.show()
    return X_mds

def module_AC_my(cluster_labels,cluster_ids,AC_conn_dict,ACP_dict):# TisaiYu[2024/9/11] 文献里直接根据ACP来算装配成本，但是ACP严格上来说是装配难度。觉得论文计算成本的方式还可以细化，从准备、作业、结果观察3个方面分别计算成本
    unique_labels = np.unique(cluster_labels)
    AC = 0
    if cluster_ids is not None:
        for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块间
            for j in range(i + 1, len(unique_labels)):
                ACij = 0
                indices_i = np.where(cluster_labels == unique_labels[i])[0]
                indices_j = np.where(cluster_labels == unique_labels[j])[0]
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        o1_partid = str(min(cluster_ids[idx_i], cluster_ids[idx_j]))
                        o2_partid = str(max(cluster_ids[idx_i], cluster_ids[idx_j]))

                        # assembly_type = AC_conn_dict.get(o1_partid,{}).get(o2_partid,None)
                        # # print(f"模块{i}和模块{j}在零部件{idx_i}和{idx_j}处以{assembly_type}连接。")
                        # if assembly_type=="Weld":
                        #     ACij+=6
                        # elif assembly_type=="Flange":
                        #     ACij+=3
                        # elif assembly_type=="Flex":
                        #     ACij+=1
                        # else:# TisaiYu[2024/8/29] None
                        #     continue
                        # 2024/9/8 使用原来的ACP，因为论文里面说模块的ACPb是只和类型相关，没说清楚呀感觉
                        ACPij = ACP_dict.get(o1_partid, {}).get(o2_partid, None)
                        if ACPij != None:
                            ACij += ACPij
                AC += ACij
    else:
        for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块间
            for j in range(i + 1, len(unique_labels)):
                ACij = 0
                indices_i = np.where(cluster_labels == unique_labels[i])[0]
                indices_j = np.where(cluster_labels == unique_labels[j])[0]
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        o1_partid = str(min(idx_i, idx_j))
                        o2_partid = str(max(idx_i, idx_j))

                        # assembly_type = AC_conn_dict.get(o1_partid,{}).get(o2_partid,None)
                        # # print(f"模块{i}和模块{j}在零部件{idx_i}和{idx_j}处以{assembly_type}连接。")
                        # if assembly_type=="Weld":
                        #     ACij+=6
                        # elif assembly_type=="Flange":
                        #     ACij+=3
                        # elif assembly_type=="Flex":
                        #     ACij+=1
                        # else:# TisaiYu[2024/8/29] None
                        #     continue
                        # 2024/9/8 使用原来的ACP，因为论文里面说模块的ACPb是只和类型相关，没说清楚呀感觉
                        ACPij = ACP_dict.get(o1_partid, {}).get(o2_partid, None)
                        if ACPij != None:
                            ACij += ACPij
                AC += ACij
    return AC
def module_AC_time_based(cluster_labels,cluster_ids,AC_conn_dict,ACP_dict):
    unique_labels = np.unique(cluster_labels)
    AC = 0
    AT=0
    if cluster_ids is not None:
        for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块间
            for j in range(i + 1, len(unique_labels)):
                ACij = 0
                indices_i = np.where(cluster_labels == unique_labels[i])[0]
                indices_j = np.where(cluster_labels == unique_labels[j])[0]
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        o1_partid = str(min(cluster_ids[idx_i], cluster_ids[idx_j]))
                        o2_partid = str(max(cluster_ids[idx_i], cluster_ids[idx_j]))
                        assembly_type = AC_conn_dict.get(o1_partid, {}).get(o2_partid, None)
                        # print(f"模块{i}和模块{j}在零部件{idx_i}和{idx_j}处以{assembly_type}连接。")
                        if assembly_type == "Weld":
                            ACij += 2.2*3
                            AT += 2.2
                        elif assembly_type == "Flange":
                            ACij += 0.875
                            AT += 0.875
                        elif assembly_type == "Flex":
                            ACij += 0.5*1
                            AT += 0.5
                        else:  # TisaiYu[2024/8/29] None
                            continue
                    AC += ACij
                AC += ACij
    else:
        for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块间
            for j in range(i + 1, len(unique_labels)):
                ACij = 0
                indices_i = np.where(cluster_labels == unique_labels[i])[0]
                indices_j = np.where(cluster_labels == unique_labels[j])[0]
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        o1_partid = str(min(idx_i, idx_j))
                        o2_partid = str(max(idx_i, idx_j))

                        assembly_type = AC_conn_dict.get(o1_partid,{}).get(o2_partid,None)
                        # print(f"模块{i}和模块{j}在零部件{idx_i}和{idx_j}处以{assembly_type}连接。")
                        if assembly_type=="Weld":
                            ACij += 2.2*2.5+0.5
                            AT += 2.2
                        elif assembly_type=="Flange":
                            ACij+=2.5*1
                            AT += 0.875
                        elif assembly_type=="Flex":
                            ACij+=0.5*1
                            AT += 0.5
                        else:# TisaiYu[2024/8/29] None
                            continue
                AC += ACij
    return AC,AT

def module_AC(cluster_labels,cluster_ids,AC_conn_dict,ACP_dict):

    unique_labels = np.unique(cluster_labels)
    AC = 0
    if cluster_ids is not None:
        for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块间
            for j in range(i + 1, len(unique_labels)):
                ACij = 0
                indices_i = np.where(cluster_labels == unique_labels[i])[0]
                indices_j = np.where(cluster_labels == unique_labels[j])[0]
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        o1_partid = str(min(cluster_ids[idx_i],cluster_ids[idx_j]))
                        o2_partid = str(max(cluster_ids[idx_i],cluster_ids[idx_j]))

                        # assembly_type = AC_conn_dict.get(o1_partid,{}).get(o2_partid,None)
                        # # print(f"模块{i}和模块{j}在零部件{idx_i}和{idx_j}处以{assembly_type}连接。")
                        # if assembly_type=="Weld":
                        #     ACij+=6
                        # elif assembly_type=="Flange":
                        #     ACij+=3
                        # elif assembly_type=="Flex":
                        #     ACij+=1
                        # else:# TisaiYu[2024/8/29] None
                        #     continue
                        # 2024/9/8 使用原来的ACP，因为论文里面说模块的ACPb是只和类型相关，没说清楚呀感觉
                        ACPij = ACP_dict.get(o1_partid,{}).get(o2_partid,None)
                        if ACPij!= None:
                            ACij += ACPij
                AC+=ACij
    else:
        for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块间
            for j in range(i + 1, len(unique_labels)):
                ACij = 0
                indices_i = np.where(cluster_labels == unique_labels[i])[0]
                indices_j = np.where(cluster_labels == unique_labels[j])[0]
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        o1_partid = str(min(idx_i, idx_j))
                        o2_partid = str(max(idx_i, idx_j))

                        # assembly_type = AC_conn_dict.get(o1_partid,{}).get(o2_partid,None)
                        # # print(f"模块{i}和模块{j}在零部件{idx_i}和{idx_j}处以{assembly_type}连接。")
                        # if assembly_type=="Weld":
                        #     ACij+=6
                        # elif assembly_type=="Flange":
                        #     ACij+=3
                        # elif assembly_type=="Flex":
                        #     ACij+=1
                        # else:# TisaiYu[2024/8/29] None
                        #     continue
                        # 2024/9/8 使用原来的ACP，因为论文里面说模块的ACPb是只和类型相关，没说清楚呀感觉
                        ACPij = ACP_dict.get(o1_partid, {}).get(o2_partid, None)
                        if ACPij != None:
                            ACij += ACPij
                AC += ACij
    return AC

def module_iterate_constraints(cluster_labels,HC_module_dict,iter_phase = False):
    unique_labels = np.unique(cluster_labels)
    part_weight_data = HC_module_dict.get("Weight")
    part_size_data = HC_module_dict.get("Size")
    module_weight = np.zeros([len(unique_labels)])  # TisaiYu[2024/8/28] 定义模块重量和尺寸的数组，后面如果要展示的话
    module_size = np.zeros([len(unique_labels), 3])  # TisaiYu[2024/8/28] length,width,height


    cl=9000 # TisaiYu[2024/8/23] 模块限长mm
    ch=3200 # TisaiYu[2024/8/23] 模块限高mm
    cw=3200 # TisaiYu[2024/8/23] 模块限宽mm
    cz=15000 # TisaiYu[2024/8/23] 模块限重kg，一个文献是500，一个是1000，这些后续都改为可输入的

    disobey_module_index = []

    for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块
        indices = np.where(cluster_labels == unique_labels[i])[0]
        minx = np.inf
        maxx = -np.inf
        miny = np.inf
        maxy = -np.inf
        minz = np.inf
        maxz = -np.inf
        # TisaiYu[2024/8/28] 先遍历一次模块内零部件求模块尺寸和重量
        for idx in indices:
            module_weight[i] += part_weight_data[idx]  # TisaiYu[2024/8/28] 模块重量
            # TisaiYu[2024/8/28] 模块尺寸
            lowx,highx,lowy,highy,lowz,highz = part_size_data[idx]
            if lowx<minx:
                minx = lowx
            if highx>maxx:
                maxx=highx
            if lowy<miny:
                miny=lowy
            if highy>maxy:
                maxy = highy
            if lowz<minz:
                minz = lowz
            if highz>maxz:
                maxz=highz
        # TisaiYu[2024/9/10] 哎，论文没说但是又的确清晰的又难想到的东西，长和宽不是按照x,y坐标计算的，而长是x,y轴计算出最大的那一个就是长，小的就是宽
        l = maxx-minx
        w = maxy-miny
        h = maxz-minz
        if w>l:
            copy = l
            l=w
            w=copy
        module_size[i, :] = [l,w,h]
        z = module_weight[i]
        if z>cz or l>cl or w>cw or h>ch:
            if len(indices)>1:# TisaiYu[2024/9/9] 只有一个组件不能再分了，即使不满足限制
                if iter_phase==True:
                    disobey_module_index.append(i)
    if len(disobey_module_index)==0:
        return module_weight,module_size
    else:
        return False,disobey_module_index

def module_light_iter(cluster_labels,HC_module_dict):
    unique_labels = np.unique(cluster_labels)
    part_weight_data = HC_module_dict.get("Weight")
    part_size_data = HC_module_dict.get("Size")
    module_weight = np.zeros([len(unique_labels)])  # TisaiYu[2024/8/28] 定义模块重量和尺寸的数组，后面如果要展示的话
    module_size = np.zeros([len(unique_labels), 3])  # TisaiYu[2024/8/28] length,width,height

    disobey_module_index = []

    for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块
        indices = np.where(cluster_labels == unique_labels[i])[0]
        minx = np.inf
        maxx = -np.inf
        miny = np.inf
        maxy = -np.inf
        minz = np.inf
        maxz = -np.inf
        # TisaiYu[2024/8/28] 先遍历一次模块内零部件求模块尺寸和重量
        for idx in indices:
            module_weight[i] += part_weight_data[idx]  # TisaiYu[2024/8/28] 模块重量
            # TisaiYu[2024/8/28] 模块尺寸
            lowx, highx, lowy, highy, lowz, highz = part_size_data[idx]
            if lowx < minx:
                minx = lowx
            if highx > maxx:
                maxx = highx
            if lowy < miny:
                miny = lowy
            if highy > maxy:
                maxy = highy
            if lowz < minz:
                minz = lowz
            if highz > maxz:
                maxz = highz
        # TisaiYu[2024/9/10] 哎，论文没说但是又的确清晰的又难想到的东西，长和宽不是按照x,y坐标计算的，而长是x,y轴计算出最大的那一个就是长，小的就是宽
        l = maxx - minx
        w = maxy - miny
        h = maxz - minz
        if w > l:
            copy = l
            l = w
            w = copy
        module_size[i, :] = [l, w, h]
        z = module_weight[i]
        if z>500: # TisaiYu[2024/9/11] 全部迭代至轻量级
            if len(indices)>1:
                disobey_module_index.append(i)
    if len(disobey_module_index) == 0:
        return module_weight, module_size
    else:
        return False, disobey_module_index
def module_parts_num_iter(cluster_labels,HC_module_dict):
    unique_labels = np.unique(cluster_labels)
    part_weight_data = HC_module_dict.get("Weight")
    part_size_data = HC_module_dict.get("Size")
    module_weight = np.zeros([len(unique_labels)])  # TisaiYu[2024/8/28] 定义模块重量和尺寸的数组，后面如果要展示的话
    module_size = np.zeros([len(unique_labels), 3])  # TisaiYu[2024/8/28] length,width,height

    disobey_module_index = []

    for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块
        indices = np.where(cluster_labels == unique_labels[i])[0]
        minx = np.inf
        maxx = -np.inf
        miny = np.inf
        maxy = -np.inf
        minz = np.inf
        maxz = -np.inf
        # TisaiYu[2024/8/28] 先遍历一次模块内零部件求模块尺寸和重量
        for idx in indices:
            module_weight[i] += part_weight_data[idx]  # TisaiYu[2024/8/28] 模块重量
            # TisaiYu[2024/8/28] 模块尺寸
            lowx, highx, lowy, highy, lowz, highz = part_size_data[idx]
            if lowx < minx:
                minx = lowx
            if highx > maxx:
                maxx = highx
            if lowy < miny:
                miny = lowy
            if highy > maxy:
                maxy = highy
            if lowz < minz:
                minz = lowz
            if highz > maxz:
                maxz = highz
        # TisaiYu[2024/9/10] 哎，论文没说但是又的确清晰的又难想到的东西，长和宽不是按照x,y坐标计算的，而长是x,y轴计算出最大的那一个就是长，小的就是宽
        l = maxx - minx
        w = maxy - miny
        h = maxz - minz
        if w > l:
            copy = l
            l = w
            w = copy
        module_size[i, :] = [l, w, h]
        z = module_weight[i]
        if z>500: # TisaiYu[2024/9/11] 全部迭代至轻量级
            if len(indices)>1:
                disobey_module_index.append(i)
    if len(disobey_module_index) == 0:
        return module_weight, module_size
    else:
        return False, disobey_module_index

def module_HC(cluster_labels,module_weight,module_size,save_excel = False):
    cl = 9000  # TisaiYu[2024/8/23] 模块限长mm
    ch = 3000  # TisaiYu[2024/8/23] 模块限高mm
    cw = 3000  # TisaiYu[2024/8/23] 模块限宽mm
    cz = 12000  # TisaiYu[2024/8/23] 模块限重kg，一个文献是500，一个是1000，这些后续都改为可输入的
    unique_labels = np.unique(cluster_labels)
    HC = 0
    fuzz_mt = FuzzySystemMT(cz,cl,cw,ch)
    MT = defaultdict(list)
    for i in range(len(unique_labels)):  # TisaiYu[2024/6/13] 遍历模块
        z = module_weight[i]
        l = module_size[i,0]
        w = module_size[i,1]
        h = module_size[i,2]
        MTi = fuzz_mt.calculate_MT(z,l,w,h)
        MT["模块编号"].append(i+1)
        MT["模块重"].append(z)
        MT["模块长"].append(l)
        MT["模块宽"].append(w)
        MT["模块高"].append(h)
        MT["MT"].append(MTi)

        if MTi==0:
            print(z,l,w,h)
        if 1<z<=500:
            MT_cost_index = 1+1*2 # TisaiYu[2024/9/23] 2个非熟练人员操作
        elif 500<z<=2500:
            MT_cost_index = 3.125+1*3+1*1.5 # TisaiYu[2024/9/23] 一个熟练人员和3个非熟练人员操作
        elif 2500<z:
            # TisaiYu[2024/9/11] 因为HC就是成本了，算的无量纲数，所以这里cost_index不采用论文的那个了，论文是先中间数，再计算实际成本，这里不要中间，直接吧无量纲映射为成本，所以去8.5
            MT_cost_index = 8.5+1.5*2+1*6 # TisaiYu[2024/9/23] 2个熟练人员和6个非熟练人员操作
        else:
            MT_cost_index = (z-12000)/100*5+8.5+1.5*2+1*6 # 2024/9/8超出限额就给很大惩罚

        cost_index_penalty_l=1
        cost_index_penalty_w=1
        cost_index_penalty_h=1
        cost_index_penalty_z=1
        if l>cl:
            cost_index_penalty_l = 1+(l-cl)/cl*2
        if w>cw:
            cost_index_penalty_w = 1+(w-cw)/cw*2
        if h>ch:
            cost_index_penalty_h =1+(h-ch)/ch*2
        if z>cz:
            cost_index_penalty_h = 1 + (z - cz) / cz
        cost_index_penalty = cost_index_penalty_l*cost_index_penalty_w*cost_index_penalty_h*cost_index_penalty_z
        HC += MTi*MT_cost_index*cost_index_penalty
    if save_excel:
        df = pd.DataFrame(MT)
        df.to_excel("MT计算结果.xlsx")
    if HC!=0:
        # print(f"在模块化方案{cluster_labels}下的HC为：{HC}")
        return HC,MT['MT']
    else:
        return -1

def module_SIC(cluster_labels,cluster_ids,AC_dict,module_weight,module_size,ACP_dict,save_MT_excel=False):
    # AC = module_AC(cluster_labels,cluster_ids,AC_dict,ACP_dict)
    AC,AT = module_AC_time_based(cluster_labels,cluster_ids,AC_dict,ACP_dict)
    if AC==-1:
        return -1
    else:
        HC,MT = module_HC(cluster_labels, module_weight,module_size,save_MT_excel)
    SIC = 3*AC+HC # TisaiYu[2024/9/11] 直接输出无量纲数，就不计算具体成本了，根据比例来，装配的费用基数1对应的是非熟练人工大概是350/天，而运输基数1对应的最轻运输工具的费用大概是340/天，所以比例大概相等，不用乘以系数

    return SIC,3*AC,HC,MT,AT


def save_module_iteration(iteration_num,cluster_labels,module_weight,moudule_size,SIC):
    unique_labels = np.unique(cluster_labels)
    iter_data = {"Iteration":["iteration"+str(iteration_num)],"SIC":[SIC]}
    iter_df = pd.DataFrame(iter_data)
    iter_df.to_excel("模块迭代结果.xlsx",header=False,index=False,mode='a')
    module_num = len(unique_labels)
    module_components=[]
    for i in range(module_num):
        components = np.where(cluster_labels==unique_labels[i])[0]
        components_solit_str = ','.join(components)
        module_components.append(components_solit_str)
    data = {"Module ID": range(1, +1), "Components":module_components,"Module Weight":module_weight,"Module Size":moudule_size}
    data_df = pd.DataFrame(data)
    data_df.to_excel("模块迭代结果.xlsx",header=False,index=False,mode='a')

