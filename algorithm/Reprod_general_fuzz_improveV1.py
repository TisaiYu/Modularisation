"""
@coding: utf-8
@File Name: Reprod
@Author: TisaiYu
@Creation Date: 2024/8/20
@version: 1.0
------------------------------------------------------------------------------------------------------------------------------------------
@Description:
二维模块划分后，通过设置的需求规则或者行业基本的标准来进行优化和继续划分
------------------------------------------------------------------------------------------------------------------------------------------
@Modification Record:
把处理改为符合甲方数据的普遍一点，特别是功能结构的处理
-----------------------------------------------------------------------------------------------------------------------------------------
文件重要修改

-------------------------------------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------------------------------

"""

import pandas as pd
import networkx as nx
from PyQt5.QtCore import Qt
from resources.db.SQLprocess import *
from collections import defaultdict
from algorithm.FuzzyAssess_ACP import *
from algorithm.FuzzyAssessMT import *
import matplotlib.pyplot as plt


class Reprod:
    def __init__(self,validate = False):
        # TisaiYu[2024/8/23] 暂时用不到
        self.query = QtSql.QSqlQuery()
        self.sql_model = QtSql.QSqlTableModel(None)
        self.sql_process = SqlProcess()
        self.validate = validate


    def connectionLayoutPlot(self):
        '''
            根据Excel的数据，绘制零部件连接网络图，对于不同的类型（三通，管道，设备采用不同的标志，如圆表示设备，三角形不是三通或者长方形来表示管道）
        :return:
        '''
        # 创建一个空的无向图
        G = nx.Graph()

        nodes = []
        edges = []
        df = pd.read_excel(f"E:\Postgraduate\YY\code\Modularization_to_python\ModularizationPy\data\example.xlsx",
                           header=0)
        for i in range(df.shape[0]):
            id = df.iat[i, 0]
            type = df.iat[i, 1]
            connpartid = df.iat[i, 2]
            nodes.append((id, {'type': type}))
            if connpartid != '/':
                edges.append((id, connpartid))  # TisaiYu[2024/8/23] nx会自动处理无向图中重复的连接。
        print(edges)
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # 定义节点的形状和颜色
        node_shapes = {
            'm1': 's',  # 长方体
            'm2': 'o',  # 圆形
            'm3': 'D',  # 正方形
        }

        node_colors = {
            'P': 'blue',
            'E': 'green',
            'V': 'purple',
        }

        # 绘制节点
        pos = nx.spring_layout(G)
        for node_type, shape in node_shapes.items():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[n for n, d in G.nodes(data=True) if d['type'] == node_type],
                node_shape=shape,
                node_color=node_colors[node_type],
                label=node_type
            )
        G.remove_edges_from(nx.selfloop_edges(G))
        # 绘制边
        nx.draw_networkx_edges(G, pos)

        # 绘制标签
        nx.draw_networkx_labels(G, pos)

        # 显示图例
        plt.legend()
        plt.show()

    def generate_DSM(self):
        pass


    def simple_dsm(self): # TisaiYu[2024/8/23] 简单根据是否连接和子系统（功能）来构建DSM看看划分效果，预期比较符合，就是功能+连接决定时。
        df = pd.read_excel(f"E:\Postgraduate\YY\code\Modularization_to_python\ModularizationPy\data\example.xlsx",
                           header=0)
        self.parts_num = len(np.unique(df.iloc[:, 0]))
        function_dict = defaultdict(list)
        connection_arr = np.zeros([self.parts_num, self.parts_num])
        function_arr = np.zeros([self.parts_num, self.parts_num])


        for i in range(df.shape[0]):
            id = df.iat[i, 0]
            type = df.iat[i, 1]
            connpartid = df.iat[i, 2]
            function = df.iat[i,4]
            if connpartid!='/':
                connection_arr[id,connpartid] = 1
            if id not in function_dict[function]:
                function_dict[function].append(id)

        for function,id_list in function_dict.items():
            for i in range(len(id_list)):
                for j in range(i + 1, len(id_list)):
                    function_arr[id_list[i], id_list[j]] = 1
                    function_arr[id_list[j], id_list[i]] = 1

        dsm = function_arr+connection_arr
        max_value = np.max(dsm)
        dsm = dsm/max_value
        for i in range(self.parts_num):
            dsm[i,i] = 1

        df = pd.DataFrame(dsm)
        df.to_excel(f"E:\Postgraduate\YY\code\Modularization_to_python\ModularizationPy\data\DSM_essay.xlsx",
                    index=False, header=False)

        # # 创建一个图形对象和两个子图
        # fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(12, 6))
        #
        # # 绘制第一个矩阵
        # cax1 = ax1.imshow(connection_arr, cmap='Blues', interpolation='none')
        # ax1.set_title('Matrix 1')
        # fig.colorbar(cax1, ax=ax1)
        #
        # # 绘制第二个矩阵
        # cax2 = ax2.imshow(function_arr, cmap='Blues', interpolation='none')
        # ax2.set_title('Matrix 2')
        # fig.colorbar(cax2, ax=ax2)
        #
        # cax3 = ax3.imshow(dsm, cmap='Blues', interpolation='none')
        # ax3.set_title('Matrix 3')
        # fig.colorbar(cax3, ax=ax3)
        # # 显示图形
        # plt.show()

    def get_conn_relationship(self,query_all_str):# TisaiYu[2024/8/23] 先通过连接关系和功能层次关系划分模块，然后得到的模块再通过三维设计后，得到连接点坐标，再通过安装成本和时间来细分模块方案。
        query = QtSql.QSqlQuery()
        query.exec_(f'''SELECT COUNT(DISTINCT PartID) FROM ({query_all_str})''') # TisaiYu[2024/8/28] 多少个要划分的零部件
        if query.next():
            self.parts_num = query.value(0)
        # TisaiYu[2024/8/23] 不加DISTINCT会重复，哎，反正是数据库的原理吧。不清楚语句哪里不对还是什么的
        query_conn = f'''SELECT DISTINCT
                o2.AssemblyType As AssemblyType,
                o1.PartID AS object1_id,
                o2.PartID AS object2_id,
                o1.Weight AS object1_weight,
                o2.Weight AS object2_weight,
                o1.Diameter AS object1_diameter,
                o2.Diameter AS object2_diameter,
                o2.z AS assembly_height
            FROM 
                ({query_all_str}) o1
            JOIN 
                ({query_all_str}) o2
            ON 
                o1.PartID = o2.ConnectionPartID
            WHERE 
                CAST(o1.PartID AS INTEGER) < CAST(o2.PartID AS INTEGER);'''
        query.exec_(query_conn)

        data = []
        while query.next():
            row = []
            for i in range(query.record().count()):
                row.append(query.value(i))
            data.append(row)
        self.numpy_array = np.array(data)
        # for i in range(self.numpy_array.shape[0]):
            # print(self.numpy_array[i,:])

    def judge_constraints(self):
        # TisaiYu[2024/8/23] 模块限制
        self.cl=4500 # TisaiYu[2024/8/23] 模块限长mm
        self.ch=2500 # TisaiYu[2024/8/23] 模块限高mm
        self.cw=2000 # TisaiYu[2024/8/23] 模块限宽mm
        self.cz=1000 # TisaiYu[2024/8/23] 模块限重kg，一个文献是500，一个是1000，这些后续都改为可输入的

        # TisaiYu[2024/8/23] 模块运输工具类别，3类，列表里分别是“轻、中等、重”，“此类下可承载重量”，“此类下资源需求指数”
        self.weight_categories = {"Category1":['light',300,1],"Category2":['light',600,3],"Category3":['light',1000,6]}

    def judge_time_and_labour_of_assembly(self):
        pass
    def judge_handlingtime_and_resource_of_equiment(self):
        pass

    def get_hierarchy_level(self,function_code):
        # 计算功能编码的层次级别，不包括根节点
        if '-' not in function_code:
            return 2  # 第一层
        else:
            # 计算'-'后面的数字部分的长度
            parts = function_code.split('-')
            if len(parts) == 2:
                return len(parts[1]) + 2  # 数字部分的长度加1
            else:
                return 2  # 默认返回第一层

    def get_common_ancestor_level(self,func1, func2):
        # 找到两个功能编码的共同祖先级别
        parts1_split = func1.split('-')
        part1_alpha = parts1_split[0]
        if len(parts1_split)>1:
            number1 = str(parts1_split[1])
        else:
            number1 = []
        parts1 = []
        parts1.append(part1_alpha)
        for i in range(len(number1)):
            parts1.append(number1[i])
        parts2_split = func2.split('-')
        part2_alpha = parts2_split[0]
        if len(parts2_split)>1:
            number2 = str(parts2_split[1])
        else:
            number2 = []
        parts2 = []
        parts2.append(part2_alpha)
        for i in range(len(number2)):
            parts2.append(number2[i])
        common_level = 1
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common_level += 1
            else:
                break
        return common_level

    def jaccard_similarity(self,set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union




    def sql_dsm(self):# TisaiYu[2024/8/23] 一个连接算一个ACPij，有多少个连接就算多少个ACPij，不清楚这个怎么对应到最后DSM上，再看看论文吧
        # TisaiYu[2024/8/23]
        DSM = np.zeros([self.parts_num,self.parts_num])

        print(DSM.shape)
        fuzzy_system_ACP = FuzzySystemACP()
        sql_model = QtSql.QSqlTableModel()
        sql_model.setTable("AddRecordTable")
        sql_model.select()
        while sql_model.canFetchMore():
            sql_model.fetchMore()
        if sql_model.data(sql_model.index(0,2))!='/':
            essay_data = True
        else:
            essay_data = False
        ACP_dict = defaultdict(list)
        ACP_value_dict = defaultdict(dict)
        for i in range(self.numpy_array.shape[0]):
            # TisaiYu[2024/8/26] Assembly cost影响因素的资源配置相关,Wcij或Rcij
            if self.numpy_array[i,0] == 'Flex':
                if float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]) <=100:
                    # print("r:",1)
                    wcij = (float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]))/100+1
                else:
                    # print("r:",2)
                    wcij = (float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4])-100)/100+2
            elif self.numpy_array[i,0] == 'Flange':
                if float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]) <= 100:
                    # print("r:",3)
                    wcij = 1.2*(float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]))/100+3
                else:
                    # print("r:",4)
                    wcij = 1.2*(float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4])-100)/100+4
            elif self.numpy_array[i,0] == 'Weld':
                if float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]) <= 100:
                    # print("r:",5)
                    wcij = 1.2*(float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]))/100+5
                else:
                    # print("r:",6)
                    wcij = 1.2*(float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4])-100)/100+6
            else:
                wcij=0

            # TisaiYu[2024/8/26] Assembly cost影响因素的装配时间相关，Tcij
            if self.numpy_array[i,0] == 'Flex':
                tcij = 1 + float(self.numpy_array[i, 5])/200
            elif self.numpy_array[i,0] == 'Flange':
                if float(self.numpy_array[i, 5])<=200:
                    # print("t:",2)
                    tcij = 2+float(self.numpy_array[i, 5])/200
                else:
                    # print("t:",3)
                    tcij = 3+float(self.numpy_array[i, 5])/200
            elif self.numpy_array[i,0] == 'Weld':
                if float(self.numpy_array[i, 5])==float(self.numpy_array[i, 6]):
                    if float(self.numpy_array[i, 5])<=200:
                        # print("t:",4)
                        tcij = 4+float(self.numpy_array[i, 5])/200
                    else:
                        # print("t:",5)
                        tcij = 5+float(self.numpy_array[i, 5])/200
                else:
                    # print("t:",6)
                    tcij = 6++max(float(self.numpy_array[i, 5]),float(self.numpy_array[i, 6]))/200
            else:
                tcij = 0
            # TisaiYu[2024/8/26] Assembly cost影响因素的返工相关，Acij
            if self.numpy_array[i,0] == 'Flex':
                if float(self.numpy_array[i, -1])<=1600:
                    # print("rw:",1)
                    acij = 1
                else:
                    # print("rw:",2)
                    acij = 2+(float(self.numpy_array[i, -1])-1600)/1600
            elif self.numpy_array[i,0] == 'Flange':
                if float(self.numpy_array[i, -1])<=1600:
                    # print("rw:",3)
                    acij =3
                else:
                    # print("rw:",4)
                    acij = 4+(float(self.numpy_array[i, -1])-1600)/1600
            elif self.numpy_array[i,0] == 'Weld':
                if float(self.numpy_array[i, -1])<=1600:
                    # print("rw",5)
                    acij = 5
                else:
                    # print("rw:",6)
                    acij = 6+(float(self.numpy_array[i, -1])-1600)/1600
            else:
                acij = 0
                # TisaiYu[2024/8/26] 不清楚文献里的T Conncetion怎么判断，所以就加了一个flex小于1600好了，似乎应该是如果装配点的坐标在最大端点坐标（连接点）和最小端点坐标（连接点）之间的话就应该是T连接，太麻烦了，暂时没管
            if wcij > 6:
                wcij = 6
            if acij > 6:
                acij = 6
            if tcij > 6:
                tcij = 6
            ACPij = fuzzy_system_ACP.calculate_ACPij(wcij, acij, tcij)
            ACP_dict["零部件1编号"].append(self.numpy_array[i,1])
            ACP_dict["零部件2编号"].append(self.numpy_array[i,2])
            ACP_dict["Wcij"].append(wcij)
            ACP_dict["Acij"].append(acij)
            ACP_dict["Tcij"].append(tcij)
            ACP_dict["ACP"].append(ACPij)
            df = pd.DataFrame(ACP_dict)
            df.to_excel("ACP计算结果.xlsx")
            ACP_value_dict[self.numpy_array[i,1]][self.numpy_array[i,2]]=ACPij
            if not essay_data:
                ACPij = 1
            DSM[int(self.numpy_array[i,1]),int(self.numpy_array[i,2])] += ACPij
        # TisaiYu[2024/8/27] 和功能相关的，这里最后自己编一个怎么根据层次功能树来得到功能的评估值，原文献是属于相同系统（功能）的+1
        function_corr_dict = defaultdict(list)
        for i in range(self.parts_num): # TisaiYu[2024/8/28] 判断功能是否相同
            for j in range(i+1,self.parts_num):
                if not essay_data:
                    function_index1 = sql_model.index(i,5)
                    function_index2 = sql_model.index(j,5)
                else:
                    function_index1 = sql_model.index(i,4)
                    function_index2 = sql_model.index(j,4)
                part_index1 = sql_model.index(i,0)
                part_index2 = sql_model.index(j,0)
                # TisaiYu[2024/8/29] 用于处理CHP1 CHP2这种，其实属于一个大系统CHP，对于本身没有数字的如function1_full_name为CWP系统，结果还是CWP
                function1_full_text = sql_model.data(function_index1,Qt.DisplayRole)
                function2_full_text = sql_model.data(function_index2,Qt.DisplayRole)
                function1_list = function1_full_text.split(',')
                function2_list = function2_full_text.split(',')
                if not essay_data:

                    association = self.calculate_association_old_example(function1_list, function2_list)

                    # TisaiYu[2024/11/13] 11.11给的系统的功能关联度，因为给的功能编码和以前不一致，后面考虑通用性封装
                    from algorithm.FHA import FHA
                    fha = FHA()
                    association = fha.func_dsm(function1_list, function2_list)

                    DSM[int(sql_model.data(part_index1,Qt.DisplayRole)),int(sql_model.data(part_index2,Qt.DisplayRole))] += association
                    function_corr_dict["零部件1编号"].append(i)
                    function_corr_dict["零部件2编号"].append(j)
                    function_corr_dict["零部件1功能"].append(','.join(function1_list))
                    function_corr_dict["零部件2功能"].append(','.join(function2_list))
                    function_corr_dict["计算关联度"].append(association)
                    print(f"零部件{i}功能{function1_list}和零部件{j}功能{function2_list}的功能关联度为：{association}")
                else:

                    if function1_full_text == function2_full_text:
                        DSM[int(sql_model.data(part_index1, Qt.DisplayRole)), int(
                            sql_model.data(part_index2, Qt.DisplayRole))] += 3
                    # else:
                    #     if function1 == function2:
                    #         DSM[int(sql_model.data(part_index1, Qt.DisplayRole)), int(
                    #             sql_model.data(part_index2, Qt.DisplayRole))] += 1.5

        df = pd.DataFrame(function_corr_dict)
        df.to_excel("功能关联度计算结果.xlsx")

        # TisaiYu[2024/8/28] 再把DSM对称填充
        for i in range(self.parts_num):
            for j in range(self.parts_num):
                if DSM[i,j] == 0 and DSM[j,i]!=0:
                    DSM[i,j] = DSM[j,i]
                elif DSM[i,j] != 0 and (DSM[j,i]==0 or DSM[j,i]<DSM[i,j]):
                    DSM[j, i] = DSM[i,j]

                else:
                    continue
        max_acp = np.max(DSM)
        for i in range(self.parts_num):
            DSM[i,i] = max_acp*1.25
        DSM = DSM/np.max(DSM)

        plt.figure()
        plt.imshow(DSM,cmap='Blues', interpolation='nearest')
        plt.show()
        from scipy.spatial.distance import pdist
        from scipy.cluster import hierarchy

        dis_mat = pdist(DSM,
                        'cosine')  # TisaiYu[2024/6/25] 层次聚类输入要么是一维数组（表示距离矩阵的压缩，比如30*30关联度矩阵，距离矩阵有30*30但是对称只取450），或者是二维数组（就是特征矩阵）

        Z = hierarchy.linkage(dis_mat, method='ward',
                              metric="cosine")  # TisaiYu[2024/9/5] 当输入是一个一维的压缩距离矩阵时（使用pdist的结果），那么metric是没有用，如果是一个二维的表示输入特征的数组，metric实际和pdist做了一样的事情


        # TisaiYu[2024/9/19] validate
        self.validate=True
        if self.validate:
            plt.figure(figsize=(10, 6))
            distances = Z[:, 2]
            reversed_last = distances[::-1]
            indexes = np.arange(1, len(distances) + 1)
            # 计算距离增长加速度,或者采用距离的最大变化
            accelerations = np.diff(reversed_last, 2)
            # accelerations = np.diff(reversed_last, 1)
            accelerations = np.pad(accelerations, (1, 1), 'constant', constant_values=(0, 0))  # 在首尾补零
            # 确定最佳聚类数
            optimal_cluster_num = np.argmax(accelerations) + 1  # +1 是因为 diff 函数减少了一个元素
            plt.figure(figsize=(10, 6))
            plt.plot(indexes, reversed_last, marker='o', label='距离')
            plt.plot(indexes, accelerations, marker='x', label='定义的加速度')
            plt.axvline(x=optimal_cluster_num, color='red', linestyle='--')
            plt.xlabel('模块数量')
            plt.ylabel('值')
            plt.title('模块设计的最佳分割点确定')
            plt.legend()
            plt.show()
            plt.figure()
            re = hierarchy.dendrogram(Z, color_threshold=Z[-(optimal_cluster_num-1),2], above_threshold_color='#bcbddc')
            plt.xticks(fontsize=12)  # 调整横坐标刻度字体大小
            plt.show()
        return DSM, self.numpy_array,ACP_value_dict,essay_data



    def judge_handling_time(self):
        pass
    def triangleFuzzFunction(self):
        pass

    def fuzz_assessment(self):
        pass
    def complete_DSM(self):
        pass

    def calculate_association_old_example(self, list1, list2):

        lf = 5
        clearness_dict = {1:1,2:0.9,3:0.7,4:0.45,5:0.2} # TisaiYu[2024/9/1] 每层功能对应的区分度，还没想好怎么用
        hirarchical_penalty_dict = {1:0.1,2:0.3,3:0.6,4:1} # TisaiYu[2024/9/1] 2个功能和父节点的层级差异惩罚
        max_num = max(len(list2),len(list1))
        min_num = min(len(list2),len(list1))

        near_rank = 0 # TisaiYu[2024/9/2] 相近平均系数，模块划分就是把功能相近的关联度拉大，功能层次差太多的拉低，这个系数记录功能组合的两两层次差异的平均值
        if list1[0] == '/' or list2[0] == '/':
            return 0.0
        proportion = self.jaccard_similarity(set(list1), set(list2))
        if proportion==1:
            return 1
        total_association = 0
        count = 0

        for func1 in list1:
            for func2 in list2:
                if func1 == func2:
                    total_association += 1.0
                else:
                    level1 = self.get_hierarchy_level(func1)
                    level2 = self.get_hierarchy_level(func2)
                    common_level = self.get_common_ancestor_level(func1, func2)
                    hirarchical_sub = max(level2-common_level,level1-common_level)
                    association = (common_level / level1+common_level/ level2)/2
                    # association = math.log(common_level + 1) / math.log(max(level1, level2) + 1)*(1-hirarchical_sub*0.1)
                    near_rank += hirarchical_sub
                    total_association += association
                count += 1
        num_sub_penalty = (max_num - min_num) / max_num
        result = total_association / count
        near_rank_average = near_rank/count

        # TisaiYu[2024/10/14] 新的功能惩罚系数
        # 定义分段函数
        if near_rank_average<np.sqrt(lf):
            near_rank_average_penalty = (0.2 / (np.sqrt(lf) - 0.5) * (near_rank_average - 0.5))
        # return FNC*(0.2 + ((1 / FNC) - 0.2) * np.power((x-np.sqrt(lf))/(lf-1-np.sqrt(lf)),k))
        else:
            near_rank_average_penalty = 0.2+(1/(num_sub_penalty+1e-3)-0.2)*np.power((near_rank_average-np.sqrt(lf))/(lf-1-np.sqrt(lf)),1)
        # TisaiYu[2024/10/14] 另外的
        b = 1/(num_sub_penalty+1e-3)
        x = near_rank_average
        p=0.8
        # 定义方程
        # def equation(k, value):
        #     return k * np.log((np.exp(1 / k) - 1) / 0.2*num_sub_penalty + 1) - value
        # # 初始猜测值
        # k_guess = 0.5
        # # 计算右边的值
        # value = ((lf-1) * p - 0.5) / (np.sqrt(lf) - 0.5)
        # # 使用牛顿迭代法求解
        # k_solution = fsolve(equation, k_guess, args=(value))
        # print("k =", k_solution[0])
        # k = k_solution[0]
        k=0.742
        near_rank_average_penalty = b * (np.exp((x - 0.5) / k * (np.sqrt(lf) - 0.5)) - 1) / (
                    np.exp(((lf - 1) * p - 0.5) / k * (np.sqrt(lf) - 0.5)) - 1)
        # proportion_penalty = -0.5*proportion+0.5
        proportion_penalty = proportion
        # num_sub_penalty= 0 # TisaiYu[2024/9/1] 由于功能数量差的惩罚
        # proportion_penalty = 0 # TisaiYu[2024/9/1] 由于交集除以并集太小的惩罚

        near_rank_average_penalty = np.exp((near_rank_average-0.5) / (1.4275*(np.sqrt(lf)-0.5))) - 1

        penalty = 1*num_sub_penalty*near_rank_average_penalty


        if count > 0:
                # result_penalty= result*(1-proportion_penalty)*(1-num_sub_penalty)*(1-(np.exp(near_rank_average/6)-1))
                # result_penalty= result

                result_penalty = result * (1 - penalty)
                if result_penalty<0:
                    return 0
                return result_penalty

        else:
            return 0.0

    def calculate_association_FBS(self,list1, list2,lf=5):

        max_num = max(len(list2),len(list1))
        min_num = min(len(list2),len(list1))

        near_rank = 0 # TisaiYu[2024/9/2] 相近平均系数，模块划分就是把功能相近的关联度拉大，功能层次差太多的拉低，这个系数记录功能组合的两两层次差异的平均值
        if list1[0] == '/' or list2[0] == '/':
            return 0.0
        proportion = self.jaccard_similarity(set(list1), set(list2))
        if proportion==1:
            return 1
        total_association = 0
        count = 0

        for func1 in list1:
            for func2 in list2:
                if func1 == func2:
                    total_association += 1.0
                else:
                    level1 = self.get_hierarchy_level(func1)
                    level2 = self.get_hierarchy_level(func2)
                    common_level = self.get_common_ancestor_level(func1, func2)
                    for i in range(common_level,level1):
                        total_association += i
                    for j in range(common_level,level2):
                        total_association += j
                count += 1
        result = total_association / count
        near_rank_average = near_rank/count
        near_rank_average_penalty = np.exp(near_rank_average/2.885)-1
        num_sub_penalty = (max_num-min_num)/max_num
        # proportion_penalty = -0.5*proportion+0.5
        proportion_penalty = proportion
        # num_sub_penalty= 0 # TisaiYu[2024/9/1] 由于功能数量差的惩罚
        # proportion_penalty = 0 # TisaiYu[2024/9/1] 由于交集除以并集太小的惩罚
        penalty = near_rank_average_penalty*num_sub_penalty
        if count > 0:
                # result_penalty= result*(1-proportion_penalty)*(1-num_sub_penalty)*(1-(np.exp(near_rank_average/6)-1))
                # result_penalty= result
                result_penalty = result * (1 - penalty)
                if result_penalty<0:
                    return 0
                return result_penalty

        else:
            return 0.0
if __name__ == "__main__":

    rep = Reprod()
    # 示例零部件功能列表
    list1 = ['FJA-200','FJB','FJA-10']

    list2 = ['FJA-200','FJB','FJA-11']

    # 计算功能列表相似度
    similarity = rep.calculate_association_old_example(list1, list2)

    print(f'零部件功能列表的关联度: {similarity}')


