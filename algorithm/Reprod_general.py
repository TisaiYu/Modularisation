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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from PyQt5 import QtSql
from PyQt5.QtCore import Qt
from db.SQLprocess import *
from collections import defaultdict
from algorithm.FuzzyAssess_ACP import *
from algorithm.FuzzyAssessMT import *
import math


class Reprod:
    def __init__(self):
        # TisaiYu[2024/8/23] 暂时用不到
        self.query = QtSql.QSqlQuery()
        self.sql_model = QtSql.QSqlTableModel(None)
        self.sql_process = SqlProcess()


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
                o1.PartID < o2.PartID;'''
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

        for i in range(self.numpy_array.shape[0]):
            # TisaiYu[2024/8/26] Assembly cost影响因素的资源配置相关,Wcij或Rcij
            if self.numpy_array[i,0] == 'Flex':
                if float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]) <=50:
                    # print("r:",1)
                    wcij = 1
                else:
                    # print("r:",2)
                    wcij = 2
            elif self.numpy_array[i,0] == 'Flange':
                if float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]) <= 50:
                    # print("r:",3)
                    wcij = 3
                else:
                    # print("r:",4)
                    wcij = 4
            elif self.numpy_array[i,0] == 'Weld':
                if float(self.numpy_array[i, 3]) + float(self.numpy_array[i, 4]) <= 50:
                    # print("r:",5)
                    wcij = 5
                else:
                    # print("r:",6)
                    wcij = 6
            else:
                wcij=0

            # TisaiYu[2024/8/26] Assembly cost影响因素的装配时间相关，Tcij
            if self.numpy_array[i,0] == 'Flex':
                # print("t:",1)
                acij = 1
            elif self.numpy_array[i,0] == 'Flange':
                if float(self.numpy_array[i, 5])<=200:
                    # print("t:",2)
                    acij = 2
                else:
                    # print("t:",3)
                    acij = 3
            elif self.numpy_array[i,0] == 'Weld':
                if float(self.numpy_array[i, 5])==float(self.numpy_array[i, 6]):
                    if float(self.numpy_array[i, 5])<=200:
                        # print("t:",4)
                        acij = 4
                    else:
                        # print("t:",5)
                        acij = 5
                else:
                    # print("t:",6)
                    acij = 6
            else:
                acij = 0
            # TisaiYu[2024/8/26] Assembly cost影响因素的返工相关，Acij
            if self.numpy_array[i,0] == 'Flex':
                if float(self.numpy_array[i, -1])<=1600:
                    # print("rw:",1)
                    tcij = 1
                else:
                    # print("rw:",2)
                    tcij = 2
            elif self.numpy_array[i,0] == 'Flange':
                if float(self.numpy_array[i, -1])<=1600:
                    # print("rw:",3)
                    tcij = 3
                else:
                    # print("rw:",4)
                    tcij = 4
            elif self.numpy_array[i,0] == 'Weld':
                if float(self.numpy_array[i, -1])<=1600:
                    # print("rw",5)
                    tcij = 5
                else:
                    # print("rw:",6)
                    tcij = 6
            else:
                tcij = 0
                # TisaiYu[2024/8/26] 不清楚文献里的T Conncetion怎么判断，所以就加了一个flex小于1600好了，似乎应该是如果装配点的坐标在最大端点坐标（连接点）和最小端点坐标（连接点）之间的话就应该是T连接，太麻烦了，暂时没管

            ACPij = fuzzy_system_ACP.calculate_ACPij(wcij, acij, tcij)
            DSM[int(self.numpy_array[i,1]),int(self.numpy_array[i,2])] = 3
        # TisaiYu[2024/8/27] 和功能相关的，这里最后自己编一个怎么根据层次功能树来得到功能的评估值，原文献是属于相同系统（功能）的+1
        sql_model = QtSql.QSqlTableModel()
        sql_model.setTable("AddRecordTable")
        sql_model.select()
        if sql_model.itemData(sql_model.index(0,1))!='/':
            essay_data = True
        else:
            essay_data = False
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
                    association = self.calculate_association(function1_list,function2_list)
                    DSM[int(sql_model.data(part_index1,Qt.DisplayRole)),int(sql_model.data(part_index2,Qt.DisplayRole))] += association
                else:
                    function1 = ''.join(filter(str.isalpha, function1_full_text))
                    function2 = ''.join(filter(str.isalpha, function2_full_text))
                    if function1_full_text == function2_full_text:
                        DSM[int(sql_model.data(part_index1, Qt.DisplayRole)), int(
                            sql_model.data(part_index2, Qt.DisplayRole))] += 1
                    else:
                        if function1 == function2:
                            DSM[int(sql_model.data(part_index1, Qt.DisplayRole)), int(
                                sql_model.data(part_index2, Qt.DisplayRole))] += 0.5


        # TisaiYu[2024/8/28] 再把DSM对称填充
        for i in range(self.parts_num): # TisaiYu[2024/8/28] 判断功能是否相同
            for j in range(self.parts_num):
                if DSM[i,j] == 0 and DSM[j,i]!=0:
                    DSM[i,j] = DSM[j,i]
                elif DSM[i,j] != 0 and (DSM[j,i]==0 or DSM[j,i]<DSM[i,j]):
                    DSM[j, i] = DSM[i,j]

                else:
                    continue
        for i in range(self.parts_num):
            DSM[i,i] = 10
        return DSM, self.numpy_array



    def judge_handling_time(self):
        pass
    def triangleFuzzFunction(self):
        pass

    def fuzz_assessment(self):
        pass
    def complete_DSM(self):
        pass

    def calculate_association(self,list1, list2):
        clearness_dict = {1:1,2:0.9,3:0.7,4:0.45,5:0.2} # TisaiYu[2024/9/1] 每层功能对应的区分度，还没想好怎么用
        hirarchical_penalty_dict = {1:0.1,2:0.3,3:0.6,4:1} # TisaiYu[2024/9/1] 2个功能和父节点的层级差异惩罚
        max_num = max(len(list2),len(list1))
        min_num = min(len(list2),len(list1))
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
                    hirarchical_sub = (level2-common_level+level1-common_level)
                    association = (common_level / level1+common_level/ level2)/2
                    # association = math.log(common_level + 1) / math.log(max(level1, level2) + 1)*(1-hirarchical_sub*0.1)
                    total_association += association
                count += 1
        result = total_association / count
        num_sub_penalty = (max_num-min_num)/max_num
        proportion_penalty = -0.5*proportion+0.5
        # num_sub_penalty= 0 # TisaiYu[2024/9/1] 由于功能数量差的惩罚
        # proportion_penalty = 0 # TisaiYu[2024/9/1] 由于交集除以并集太小的惩罚
        if count > 0:
            return result*(1-proportion_penalty)*(1-num_sub_penalty)
        else:
            return 0.0


if __name__ == "__main__":

    rep = Reprod()
    # 示例零部件功能列表
    list1 = ['fja-200','fjb']
    list2 = ['fja-201','fja-202']

    # 计算功能列表相似度
    similarity = rep.calculate_association(list1, list2)
    print(f'零部件功能列表的关联度: {similarity}')


