import time

import numpy as np

from algorithm.ClusterOptimize import *
from PyQt5.QtCore import QThread,pyqtSignal,pyqtSlot,QObject,QMutex
"""
@coding: utf-8
@File Name: HierarchicalClusteringV3
@Author: TisaiYu
@Creation Date: 2024/8/28
@version: 1.0
------------------------------------------------------------------------------------------------------------------------------------------
@Description: 
添加了计算三维的装配成本和运输时间以及最后安装成本的算法
------------------------------------------------------------------------------------------------------------------------------------------
@Modification Record: 

-----------------------------------------------------------------------------------------------------------------------------------------
文件重要修改

-------------------------------------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------------------------------

"""




class HierarchicalClustering(QThread):
    clustering_finished = pyqtSignal(np.ndarray, float,str,dict,int,int) # TisaiYu[2024/6/20] 信号定义为类的属性，而不是实例属性，是因为连接要写在实例化调用函数之前，这样实例化的函数发出信号才可以被响应
    divide_dedrogram = pyqtSignal(np.ndarray) # TisaiYu[2024/6/20] 信号定义为类的属性，而不是实例属性，是因为连接要写在实例化调用函数之前，这样实例化的函数发出信号才可以被响应
    progress_sig = pyqtSignal()
    def __init__(self,DSM,Z,metric_name,thread_id,AC_dict=None,HC_dict=None,ACP_dict=None):
        super(HierarchicalClustering, self).__init__()
        self.DSM = DSM
        self.metric_name = metric_name
        self.Z = None
        self.metric_values = None
        self.thread_id = thread_id
        self.Z =Z
        self.AC_dict = AC_dict
        self.ACP_dict = ACP_dict
        self.HC_dict = HC_dict

    def run(self):

        func = self.compute_by_which(self.metric_name)
        if self.metric_name!='SIC':
            best_la, best_value, values_dict,iter_constraints_module_num = func(self.Z, self.DSM)
        else:
            best_la,best_value,values_dict,iter_constraints_module_num = func(self.Z,self.DSM)
        compute_dict = {"CE": compute_CE, "Sil": compute_Sil, "G": compute_G,"Q":compute_Q}
        # TisaiYu[2024/6/25] 层次聚类的结果添加微小扰动，看遗传模拟退火能不能回来
        # best_cluster_labelss = np.array(
        #     [3, 4, 3, 3, 3, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 7, 6, 6])
        # sqrt_num = 2
        # num_original_classes = len(np.unique(best_cluster_labels))
        # min_classes = max(1, num_original_classes - sqrt_num)
        # max_classes = num_original_classes + sqrt_num
        # new_labels = np.random.randint(-sqrt_num, sqrt_num, size=len(best_cluster_labels))
        # change_mask = np.random.rand(new_labels.size) < 0.3
        # new_labels[change_mask] = 0
        # new_labels = best_cluster_labels + new_labels
        # new_labels[np.where(new_labels < 1)] = 1
        # unique_labels = np.unique(new_labels)
        # if len(unique_labels) >= min_classes and len(unique_labels) <= max_classes:
        #     # Relabel to ensure continuity from 1 to len(unique_labels)
        #     label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, 1)}
        #     best_cluster_labels = np.array([label_mapping[label] for label in new_labels])

        # hgsa = GeneticSimulatedAnnealing(best_la, self.DSM, compute_Q)
        # best_individual, best_fitness = hgsa.optimize()
        # print(f"{self.metric_name}遗传模拟退火优化后的fitness：", best_fitness)
        # print(f"{self.metric_name}遗传模拟退火优化输入方案：", best_la)
        # print(f"{self.metric_name}遗传模拟退火优化后的最佳分类方案：", best_individual)
        # print(f"{self.metric_name}遗传模拟退火优化后方案差异：", np.array(best_la) - np.array(best_individual))
        self.clustering_finished.emit(best_la,best_value,self.metric_name,values_dict,self.thread_id,iter_constraints_module_num)


    def compute_by_which(self,metric_name):
        func = None
        if metric_name == "CE":
            func = self.find_best_clustering_by_CE
        if metric_name == "Sil":
            func= self.find_best_clustering_by_Sil
        if metric_name == "DB":
            func=self.find_best_clustering_by_DB
        if metric_name == "D1D2":
            func=self.find_best_clustering_by_D1D2
        if metric_name == "CH":
            func=self.find_best_clustering_by_CH
        if metric_name == "G":
            func = self.find_best_clustering_by_G
        if metric_name == "Gap":
            func = self.find_best_clustering_by_Gap
        if metric_name == "Q":
            func = self.find_best_clustering_by_Q
        if metric_name == "SIC":
            func = self.find_best_clustering_by_SIC
        return func


    def find_best_clustering_by_D1D2(self,Z, dist_matrix): # TisaiYu[2024/6/4] 这个指标不行，可能代码写的有问题或者理解有误。
        max_D1 = -np.inf
        min_D2 = np.inf
        best_cluster_labels = None
        D1s = []
        D2s = []
        D1D2S = []
        best_index = 0
        for num in range(len(Z)):
            cluster_labels = hierarchy.fcluster(Z, num+1, criterion='maxclust')
            D1 = time_count(compute_D1,cluster_labels, dist_matrix)
            D2 = time_count(compute_D2,cluster_labels, dist_matrix)
            if D1 > max_D1 and D2 < min_D2: # TisaiYu[2024/6/4] 这个判断逻辑太苛刻了，导致错误。可能D1比上次大很多，但是D2没减小。
                max_D1 = D1
                min_D2 = D2
                best_cluster_labels = cluster_labels
                best_index = num
            # print(f"D1:{D1}, D2:{D2}:")
            # print(cluster_labels)
            if D2==0:
                D1s.append(D1)
                D2s.append(D2)
                D1D2S.append(D2)
            else:
                D1D2S.append(D1/D2)
                D1s.append(D1)
                D2s.append(D2)
            self.progress_sig.emit()
        # plt.figure()
        # plt.plot(D1s)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the D1s')
        # plt.show()
        # plt.figure()
        # plt.plot(D2s)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the D2s')
        # plt.show()
        if min_D2!=0:
            return best_cluster_labels, max_D1/min_D2,D1D2S,best_index
        else:
            return best_cluster_labels,0,D1D2S,best_index


    def find_best_clustering_by_CE(self,Z, dist_matrix):
        max_ce = 0
        best_cluster_labels = None
        ces = {}

        best_index = 0
        for num in range(2, len(Z)):
            cluster_labels = hierarchy.fcluster(Z, num, criterion='maxclust')
            ce = time_count(compute_CE, cluster_labels, dist_matrix)
            if ce > max_ce:
                best_index = num
                max_ce = ce
                best_cluster_labels = cluster_labels
            # print("CE（越大越好）:",ce)
            # print("此CE下的聚类结果：",cluster_labels)
            ces[num] = ce

            # TisaiYu[2024/8/28] 根据数据库的内容计算模块方案的成本指数算法

            self.progress_sig.emit()
        # plt.figure()
        # plt.plot(ces)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the CE')
        # plt.show()
        return best_cluster_labels, max_ce,ces,best_index

    def find_best_clustering_by_G(self, Z,dist_matrix):
        max_G=0
        Gs={}
        best_cluster_labels = None
        best_index = 0
        for num in range(2, len(Z)):
            cluster_labels = hierarchy.fcluster(Z, num, criterion='maxclust')
            G = time_count(compute_G, cluster_labels, dist_matrix)
            if G>max_G:
                best_index = num
                max_G = G
                best_cluster_labels = cluster_labels
            Gs[num] =G
            self.progress_sig.emit()
        # plt.figure()
        # plt.plot(Gs)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the G')
        # plt.show()
        # print(max_G)
        # return best_cluster_labels,max_G,Gs,best_index

        return best_cluster_labels,max_G,Gs,best_index

    def module_divid_elbow(self,Z,light_begin=False):
        if light_begin:
            distances = Z[-3:,2]
        else:
            distances = Z[:, 2]
        reverse_distance = distances[::-1]
        distances_change = -np.diff(reverse_distance, prepend=reverse_distance[0])
        module_optimum_num = np.argmax(distances_change) + 1  # TisaiYu[2024/9/10] 返回了下标，但是模块数是从1开始的，所以+1

        cluster_labels = hierarchy.fcluster(Z, module_optimum_num, criterion='maxclust')

        i=0
        while len(np.unique(cluster_labels)) != (module_optimum_num+i):
            i+=1
            print("module_optimum_num!=cluster_labels_nums")
            distance = float(Z[-(module_optimum_num+i - 1), 2]) - 0.01
            cluster_labels = hierarchy.fcluster(Z, distance, criterion='distance')


        best_cluster_labels = cluster_labels
        self.progress_sig.emit()
        return best_cluster_labels

    def iter_final(self,cluster_labels,SICs):  # TisaiYu[2024/9/12] 这个太花时间了，验证报告可以有后面的iter_final，说明后面成本会上升很快，因为轻量级了都，再细分，需要的轻量级设备就多了。模块结果时就不需要跑这个了
        iteration = 1
        unique_labels = np.unique(cluster_labels)
        parts_num = len(cluster_labels)
        nums = len(unique_labels)
        while nums!=parts_num:
            new_labels = None
            for index in unique_labels:
                comps = np.where(cluster_labels==index)[0]
                if len(comps)>1:
                    indices = np.ix_(comps,comps)
                    DSM = self.DSM[indices]
                    Z_new = hierarchy.linkage(DSM, method='weighted', metric="cosine")
                    small_best_clustering_labels = self.module_divid_elbow(Z_new, False)
                    cluster_labels[comps] = small_best_clustering_labels + cluster_labels.max()  # TisaiYu[2024/8/30] 确保标签位移，就算不是连续也不影响分类
                    unique_labels_new = np.unique(cluster_labels)
                    # 创建一个字典，将原始标签映射到新的连续标签
                    label_mapping = {old_label: new_label for new_label, old_label in
                                     enumerate(unique_labels_new, start=1)}
                    copy_labels = np.copy(cluster_labels)
                    # 使用字典进行标签转换
                    new_labels = np.array([label_mapping[label] for label in copy_labels])
                    module_weight, module_size = module_iterate_constraints(new_labels, self.HC_dict,
                                                                        iter_phase=False)  # TisaiYu[2024/8/29] iter_phase=False表示还在第一次iteration
                    SIC, corr_AC, corr_HC = module_SIC(cluster_labels, None, self.AC_dict, module_weight, module_size,
                                                       self.ACP_dict)
                    print(":SIC:", SIC)
                    print(":AC:", corr_AC)
                    print(":HC:", corr_HC)
                    print(f"iteration{iteration}")
                    print(
                        '**********************************************************************************************')
                    for i in np.unique(new_labels):
                        print(f"M{i}:{np.where(new_labels == i)[0]}")
                    print(
                        '**********************************************************************************************')
                    iteration+=1
                else:
                    continue
            cluster_labels=new_labels
            nums = len(np.unique(cluster_labels))
        return SICs
    def find_best_clustering_by_SIC(self, Z,DSM,till_parts_num=True):
        SICs={}
        best_cluster_labels = self.module_divid_elbow(Z)
        module_weight, module_size = module_iterate_constraints(best_cluster_labels, self.HC_dict,iter_phase=False)  # TisaiYu[2024/8/29] iter_phase=False表示还在第一次iteration
        min_SIC1, corr_AC, corr_HC = module_SIC(best_cluster_labels, None, self.AC_dict, module_weight, module_size, self.ACP_dict)
        print(":SIC:",min_SIC1)
        print(":AC:",corr_AC)
        print(":HC:",corr_HC)
        SICs[len(np.unique(best_cluster_labels))] = min_SIC1
        best_cluster_labels,new_SICS,iter_constraints_module_num = self.iter_then(best_cluster_labels,SICs)
        if till_parts_num:
            final_SICs = self.iter_final(best_cluster_labels, new_SICS)
            min_SIC=min(final_SICs.values())
        else:
            min_SIC = min(new_SICS.values())
        self.progress_sig.emit()

        return best_cluster_labels,min_SIC,new_SICS,iter_constraints_module_num

    def iter_then(self,cluster_labels,SICs,only_final_result=False,iter_light=True):
    # TisaiYu[2024/8/30] 说明模块有不符合约束的，要进行迭代，把不符合约束的模块的零部件的DSM提取出来，再进行模块划分
        begin_light_iter = False
        iteration = 1
        iter_constraints_module_num = 0
        if not only_final_result:
            print(f"iteration{iteration}")
            print('**********************************************************************************************')
            for i in np.unique(cluster_labels):
                print(f"M{i}:{np.where(cluster_labels == i)[0]}")
            print('**********************************************************************************************')

        last_iter_module_num = 0
        while True:
            if_iteration = module_iterate_constraints(cluster_labels, self.HC_dict, iter_phase=True)
            if not isinstance(if_iteration[0],bool):
                if iter_constraints_module_num==0:
                    iter_constraints_module_num=len(np.unique(cluster_labels))
                if not iter_light:
                    return cluster_labels,SICs,iter_constraints_module_num
                if_iteration_light = module_light_iter(cluster_labels,self.HC_dict)
                if not isinstance(if_iteration_light[0], bool):
                    return cluster_labels,SICs,iter_constraints_module_num
                disobey_module_list = if_iteration_light[1]
                begin_light_iter=True
            else:
                disobey_module_list = if_iteration[1]
            new_labels= None
            unique_labels = np.unique(cluster_labels)

            for module_index in disobey_module_list:
                disobey_module_comps = np.where(cluster_labels==unique_labels[module_index])[0]
                module_comps_indices = np.ix_(disobey_module_comps,disobey_module_comps)
                DSM = self.DSM[module_comps_indices]
                Z_new = hierarchy.linkage(DSM, method='weighted', metric="cosine") # TisaiYu[2024/9/12] 第一次划分应该使用ward距离识别功能决定的大部分片区域，然后使用single（最小距离）相当于识别模块间的最优接口，应该是簇间距离最小的那两个
                self.divide_dedrogram.emit(Z_new)
                # small_best_clustering_labels,min_SIC_divide_,AC_,HC_, best_index_devide_= self.module_divide(Z_new,disobey_module_comps)
                small_best_clustering_labels= self.module_divid_elbow(Z_new,begin_light_iter)
                cluster_labels[disobey_module_comps] = small_best_clustering_labels + cluster_labels.max()# TisaiYu[2024/8/30] 确保标签位移，就算不是连续也不影响分类
                unique_labels_new = np.unique(cluster_labels)
                # 创建一个字典，将原始标签映射到新的连续标签
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels_new, start=1)}
                copy_labels = np.copy(cluster_labels)
                # 使用字典进行标签转换
                new_labels = np.array([label_mapping[label] for label in copy_labels])
                this_iter_module_num =  len(np.unique(new_labels))
                if this_iter_module_num==last_iter_module_num:
                    continue
                module_weight, module_size = module_iterate_constraints(new_labels, self.HC_dict,iter_phase=False)  # TisaiYu[2024/8/29] iter_phase=False表示还在第一次iteration
                iteration += 1
                SIC,corr_AC,corr_HC = module_SIC(new_labels, None, self.AC_dict, module_weight, module_size, self.ACP_dict)
                SICs[this_iter_module_num] = SIC
                print(":SIC:",SIC)
                print(":AC:", corr_AC)
                print(":HC:", corr_HC)
                if not only_final_result:
                    print(f"iteration{iteration}")
                    print('**********************************************************************************************')
                    for i in np.unique(new_labels):
                        print(f"M{i}:{np.where(new_labels == i)[0]}")
                    print('**********************************************************************************************')
                last_iter_module_num = this_iter_module_num
            cluster_labels = new_labels
            last_iter_module_num = len(np.unique(cluster_labels))




    def find_best_clustering_by_Q(self, Z, dist_matrix): # TisaiYu[2024/6/19] 这个计算有问题反正，单调的那种
        max_Q = -np.inf
        best_cluster_labels = None
        Qs={}

        best_index = 0
        for num in range(2, len(Z)):# TisaiYu[2024/8/29] for num in range(1, len(Z) + 1)
            cluster_labels = hierarchy.fcluster(Z, num, criterion='maxclust')
            Q = time_count(compute_Q, cluster_labels, dist_matrix)
            if Q > max_Q:
                best_index = num
                max_Q = Q
                best_cluster_labels = cluster_labels
            # print(Q)
            # print(cluster_labels)
            Qs[num] = Q

            self.progress_sig.emit()
        # plt.figure()
        # plt.plot(Qs)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the Q')
        # plt.show()
        return best_cluster_labels, max_Q,Qs,best_index

    def find_best_clustering_by_CH(self, Z, dist_matrix):
        max_CH = -np.inf
        best_cluster_labels = None
        CHs={}
        SICs = []
        best_index = 0
        for num in range(2, len(Z)):
            cluster_labels = hierarchy.fcluster(Z, num, criterion='maxclust')
            CH = time_count(compute_CH, cluster_labels, dist_matrix)
            if CH > max_CH:
                best_index = num
                max_CH = CH
                best_cluster_labels = cluster_labels
            # print(Q)
            # print(cluster_labels)
            CHs[num]=CH
            # if module_iterate_constraints(cluster_labels, self.HC_dict):
            #     SIC = module_SIC(cluster_labels, self.AC_dict, self.HC_dict)
            self.progress_sig.emit()
        # plt.figure()
        # plt.plot(list(CHs.values()))
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the CH')
        # plt.show()
        return best_cluster_labels, max_CH,CHs,best_index

    def find_best_clustering_by_DB(self, Z, dist_matrix):
        min_DB = np.inf
        best_cluster_labels = None
        DBs={}

        best_index = 0
        for num in range(2, len(Z)):
            cluster_labels = hierarchy.fcluster(Z, num, criterion='maxclust')
            DB = time_count(compute_DB, cluster_labels, dist_matrix)
            if DB < min_DB and DB >0:
                best_index = num
                min_DB = DB
                best_cluster_labels = cluster_labels
            # print(Q)
            # print(cluster_labels)
            DBs[num]=DB

            self.progress_sig.emit()
        # plt.figure()
        # plt.plot(DBs)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the DB')
        # plt.show()
        return best_cluster_labels, min_DB,DBs,best_index

    def find_best_clustering_by_Sil(self, Z, dist_matrix):
        max_Sil = -1 # TisaiYu[2024/6/5] 因为Sil是-1到1，越大越好
        best_cluster_labels = None
        Sils = {}
        best_index = 0
        for num in range(2, len(Z)):
            cluster_labels = hierarchy.fcluster(Z, num, criterion='maxclust')
            Sil = time_count(compute_Sil, cluster_labels, dist_matrix)
            if Sil > max_Sil:
                best_index = num
                max_Sil = Sil
                best_cluster_labels = cluster_labels
            # print(Q)
            # print(cluster_labels)
            Sils[num] = Sil
            # if module_iterate_constraints(cluster_labels, self.HC_dict):
            #     SIC = module_SIC(cluster_labels, self.AC_dict, self.HC_dict)
            self.progress_sig.emit()
        # plt.figure()
        # plt.plot(Sils)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the Sil')
        # plt.show()
        return best_cluster_labels, max_Sil,Sils,best_index

    def find_best_clustering_by_Gap(self, Z, dist_matrix):
        max_Gap = -np.inf # TisaiYu[2024/6/5] 因为Sil是-1到1，越大越好
        best_cluster_labels = None
        Gaps = []

        best_index = 0
        for num in range(len(Z)):
            cluster_labels = hierarchy.fcluster(Z, num+1, criterion='maxclust')

            Gap = time_count(compute_Gap,cluster_labels, dist_matrix)
            if Gap > max_Gap:
                best_index = num
                max_Gap = Gap
                best_cluster_labels = cluster_labels
            # print(Q)
            # print(cluster_labels)
            Gaps.append(Gap)
            # if module_iterate_constraints(cluster_labels, self.HC_dict):
            #     SIC = module_SIC(cluster_labels, self.AC_dict, self.HC_dict)
            self.progress_sig.emit()
        # plt.figurefinished_print
        # plt.plot(Gaps)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Curve of the Gap')
        # plt.show()
        return best_cluster_labels, max_Gap,Gaps,best_index


