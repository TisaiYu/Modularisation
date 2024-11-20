import pandas as pd
import numpy as np
from itertools import product

class FHA:
    def __init__(self):
        self.description  = """
S:=S1,S2,S3,S4,S5,S6
S1:=S11
S2:=S21,S22,S23,S24
S3:=S31,S32,S33,S34,S35
S4:=S41,S42
S5:=S51
S6:=S61,S62,S63
S61:=S611,S612
    """
        # self.description = """
        # FJA-31:=FJA-310,FJA-311,FJA-312
        # FJA-30:=FJA-300,FJA-301,FJA-302
        # FJA-21:=FJA-210,FJA-211,FJA-212
        # FJA-20:=FJA-200,FJA-201,FJA-202,FJA-203
        # FJA-3:=FJA-30,FJA-31
        # FJA-2:=FJA-20,FJA-21
        # FJA-1:=FJA-10,FJA-11
        # FJA:=FJA-1,FJA-2,FJA-3
        # FJ:=FJA,FJB
        # """
        self.tree_structure = self.parse_tree_description(self.description)
        self.level_dict = {}
        self.parent_dict = {}
        self.leaf_nodes = set()
        self.calculate_levels_and_parents(self.get_root(), 1, None)

    def parse_tree_description(self, description):
        tree_structure = {}
        for line in description.strip().split("\n"):
            if ':=' in line:
                parent, children = line.split(":=")
                children = [child.strip() for child in children.split(",")]
                tree_structure[parent.strip()] = children
        return tree_structure

    def get_root(self):
        # 找到根节点，即没有任何父节点的节点
        children_set = {child for children in self.tree_structure.values() for child in children}
        for node in self.tree_structure:
            if node not in children_set:
                return node

    def calculate_levels_and_parents(self, node, level, parent):
        self.level_dict[node] = level
        self.parent_dict[node] = parent
        if node in self.tree_structure:
            for child in self.tree_structure[node]:
                self.calculate_levels_and_parents(child, level + 1, node)
        else:
            self.leaf_nodes.add(node)

    def find_common_parent(self, node1, node2):
        ancestors1 = set()
        while node1:
            ancestors1.add(node1)
            node1 = self.parent_dict.get(node1)
        while node2:
            if node2 in ancestors1:
                return node2
            node2 = self.parent_dict.get(node2)
        return None

    def generate_combinations_with_parents(self, a1, a2):
        combinations = list(product(a1, a2))
        combination_levels = []
        for item1, item2 in combinations:
            level1 = self.level_dict.get(item1, 'Unknown')
            level2 = self.level_dict.get(item2, 'Unknown')
            common_parent = self.find_common_parent(item1, item2)
            common_parent_level = self.level_dict.get(common_parent, 'Unknown')
            combination_levels.append((item1, item2, level1, level2, common_parent, common_parent_level))
        return combination_levels

    def jaccard_similarity(self,set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def func_dsm(self,list1,list2):
        # 生成组合并计算层级及共同父节点层级

        lf = 4
        max_num = max(len(list2), len(list1))
        min_num = min(len(list2), len(list1))

        near_rank = 0  # TisaiYu[2024/9/2] 相近平均系数，模块划分就是把功能相近的关联度拉大，功能层次差太多的拉低，这个系数记录功能组合的两两层次差异的平均值
        if list1[0] == '/' or list2[0] == '/':
            return 0.0
        proportion = self.jaccard_similarity(set(list1), set(list2))
        if proportion == 1:
            return 1
        total_association = 0
        count = 0
        combinations = list(product(list1, list2))
        for func1, func2 in combinations:
            if func1 == func2:
                total_association += 1.0
            else:
                level1 = self.level_dict.get(func1, 'Unknown')
                level2 = self.level_dict.get(func2, 'Unknown')
                common_parent = self.find_common_parent(func1, func2)
                common_level = self.level_dict.get(common_parent, 'Unknown')
                hirarchical_sub = max(level2 - common_level, level1 - common_level)
                association = (common_level / level1 + common_level / level2) / 2
                near_rank += hirarchical_sub
                total_association += association
            count += 1
        num_sub_penalty = (max_num - min_num) / max_num
        result = total_association / count
        near_rank_average = near_rank / count

        near_rank_average_penalty = np.exp((near_rank_average - 0.5) / (1.4275 * (np.sqrt(lf) - 0.5))) - 1

        penalty = 1 * num_sub_penalty * near_rank_average_penalty
        if count > 0:
                # result_penalty= result*(1-proportion_penalty)*(1-num_sub_penalty)*(1-(np.exp(near_rank_average/6)-1))
                # result_penalty= result

                result_penalty = result * (1 - penalty)
                if result_penalty<0:
                    return 0
                return result_penalty
        else:
            return 0.0

    def func_dsm_distance(self,list1,list2):
        # 生成组合并计算层级及共同父节点层级



        near_rank = 0  # TisaiYu[2024/9/2] 相近平均系数，模块划分就是把功能相近的关联度拉大，功能层次差太多的拉低，这个系数记录功能组合的两两层次差异的平均值
        if list1[0] == '/' or list2[0] == '/':
            return -1
        proportion = self.jaccard_similarity(set(list1), set(list2))
        if proportion == 1:
            return 0
        total_distance = 0
        count = 0
        combinations = list(product(list1, list2))

        # TisaiYu[2024/11/13] 记录折合因子，如果两个功能对比时，发现它们的层数情况（以及父节点）在之前的几对功能里已经出现了相同的，则要进行折合加权（以此来消除以前功能数量惩罚，同时也不需要层次惩罚了）
        function_pair_level_record = []

        for func1, func2 in combinations:
            if func1 == func2:
                total_distance += 0
            else:
                level1 = self.level_dict.get(func1, 'Unknown')
                level2 = self.level_dict.get(func2, 'Unknown')
                common_parent = self.find_common_parent(func1, func2)
                common_level = self.level_dict.get(common_parent, 'Unknown')
                distance = level2 + level1 - common_level
                if_same_add_already_unique_1 = (level1,level2,common_level)
                if_same_add_already_unique_2 = (level2,level1,common_level)
                if if_same_add_already_unique_1 not in function_pair_level_record and if_same_add_already_unique_2 not in function_pair_level_record:
                    function_pair_level_record.append(if_same_add_already_unique_1)
                else:
                    distance = distance*1.05

                total_distance += distance
            count += 1

        result = total_distance / count


        if count > 0:
                return result
        else:
            return 0.0
if __name__ == "__main__":


    # 创建TreeStructure实例
    fha = FHA()

    # 示例输入列表
    a1 = ['S24']
    a2 = ['S611','S62']
    d = fha.func_dsm_distance(a1,a2)
    print(d)
    a1 = ['S611','S62']
    a2 = ['S611','S62']
    d = fha.func_dsm_distance(a1,a2)
    print(d)

