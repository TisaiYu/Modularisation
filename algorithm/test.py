import pandas as pd

# 解析树结构描述，构建树结构字典
def parse_tree_description(description):
    tree_structure = {}
    for line in description.strip().split("\n"):
        if ':=' in line:
            parent, children = line.split(":=")
            tree_structure[parent.strip()] = [child.strip() for child in children.split(",")]
    return tree_structure

# 递归函数计算每个节点的层级
def calculate_levels(node, level, tree_structure, level_dict):
    level_dict[node] = level
    if node in tree_structure:
        for child in tree_structure[node]:
            calculate_levels(child, level + 1, tree_structure, level_dict)

# 示例树结构描述
tree_description = """
S:=S1,S2,S3,S4,S5,S6
S1:=S11
S2:=S21,S22,S23,S24
S3:=S31,S32,S33
S4:=S41,S42
S5:=S51
S6:=S61,S62,S63
S61:=S611,S612
"""

# 解析树结构并计算层级
tree_structure = parse_tree_description(tree_description)
level_dict = {}
calculate_levels('S', 1, tree_structure, level_dict)

# 打印层级字典供查看
print(level_dict)

# 示例输入列表
a1 = ['S1', 'S2', 'S3']
a2 = ['S1', 'S2']

# 生成两两组合并计算层级
from itertools import product

def generate_combinations(a1, a2, level_dict):
    combinations = list(product(a1, a2))
    combination_levels = [(item1, item2, level_dict.get(item1, 'Unknown'), level_dict.get(item2, 'Unknown')) for item1, item2 in combinations]
    return combination_levels

# 生成组合并计算层级
combinations_with_levels = generate_combinations(a1, a2, level_dict)

# 打印结果
for combo in combinations_with_levels:
    print(f"组合: {combo[0]} 和 {combo[1]}，层级: {combo[2]} 和 {combo[3]}")
