import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt


def example_system_function_parse():
    """
    示例系统的功能解析
    :return:
    """
    # 功能描述字符串
    description = """
    # 索引预定义
    g1 := 1,2,3,4
    g2 := 4,5,6,7,8,9,10
    g3 := 10,11,12
    g4 := 12,13,14,15,16,17
    g5 := 16,17,18,19,20,21
    g6 := 21,24,27
    g7 := 21,22,25,28
    g8 := 21,22,23,26,29
    g9 := 27,31
    g10 := 28,39,31
    g11 := 29,30,31
    g12 := 31,32,33,34,16
    g13 := 16,35,63,60,36
    g14 := 36,37,38
    g15 := 18,34
    g16 := 32,31
    g17 := 23,48,49,65
    g18 := 57,56,54
    g19 := 54,55,52,47,45
    g20 := 45,40,27
    g21 := 45,41,28
    g22 := 45,42,29
    g23 := 54,53,51,50,48
    g24 := 45,46,65
    g25 := 12,64
    g26 := 11,58,59,60
    g27 := 14,61,62,63
    
    # 功能定义
    FJA-10 := g1
    FJA-11 := g2
    FJA-200 := g1,g2,g3,g4
    FJA-201 := g5,g6,g7,g8
    FJA-202 := g9,g10,g11,g12
    FJA-203 := g13,g14
    FJA-210 := g15
    FJA-211 := g27
    FJA-212 := g26
    FJA-300 := g6,g9
    FJA-301 := g7,g10
    FJA-302 := g8,g11
    FJA-310 := g16,g6,g9,g7,g10,g8,g11,g17
    FJA-311 := g18,g19,g20,g21,g22,g6,g7,g8,g17
    FJA-312 := g18,g23,g6,g7,g8,g9,g10,11,g20,g21,g22,g24
    FJB := g1,g2,g3,g25
    
    # 功能树
    FJA-31 := FJA-310,FJA-311,FJA-312
    FJA-30 := FJA-300,FJA-301,FJA-302
    FJA-21 := FJA-210,FJA-211,FJA-212
    FJA-20 := FJA-200,FJA-201,FJA-202,FJA-203
    FJA-3 := FJA-30,FJA-31
    FJA-2 := FJA-20,FJA-21
    FJA-1 := FJA-10,FJA-11
    FJA := FJA-1,FJ-2,FJ-3
    FJ := FJA,FJB
    """

    # 解析功能描述
    part_function_map = {}

    # 解析 g 组合
    g_map = {}

    # 解析 g 组合
    for line in description.splitlines():
        line = line.strip()
        if ':=' in line:
            if line.startswith('g'):
                group, parts = line.split(' := ')
                g_map[group] = parts.split(',')

    # 解析功能定义
    for line in description.splitlines():
        line = line.strip()
        if ':=' in line and line.startswith('F'):
            function, groups = line.split(' := ')
            groups = groups.split(',')
            for group in groups:
                group = group.strip()
                if group in g_map:
                    ids = g_map[group]
                    for id in ids:
                        id = id.strip()
                        if id not in part_function_map:
                            part_function_map[id] = set()  # 使用集合来避免重复
                        part_function_map[id].add(function)  # 添加功能

    # 创建完整的 ID 列表
    max_id = 64  # 假设最大 ID 为 64
    all_ids = {str(i): [] for i in range(1, max_id + 1)}

    # 填充功能映射
    for part_id, functions in part_function_map.items():
        all_ids[part_id] = list(functions)  # 转换为列表

    # 构建 DataFrame
    data = []
    for part_id, functions in all_ids.items():
        data.append({'ID': part_id, 'Functions': ','.join(functions) if functions else '/'})

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 按 ID 排序
    df['ID'] = df['ID'].astype(int) - 1  # 转换为整数类型以便排序
    df = df.sort_values(by='ID')

    # 保存为 Excel 文件
    df.to_excel('part_function_mapping.xlsx', index=False)

    print("Excel 文件已生成：part_function_mapping.xlsx")

def draw_and_save_function_tree(function_node_edge_list,save_tree_pic=True):
    # 创建一个空的有向图
    G = nx.DiGraph()

    # 添加节点和边
    G.add_edges_from(function_node_edge_list)

def system_function_parse(system_id : str = 0):
    """
    看新给的系统的txt文件功能好像是生成的不是写的，所以后续可能都是这样的编号吧，就写个通用的吧，不然每次txt文件都要根据内容写个解析。
    :return:
    """

    function_tree_description_lines = []
    function_assembly_description_lines = []
    assembly_part_description_lines = []
    with open("systemtxt/11.11system/function.txt") as f:
        for line in f:
            if line.strip():
                line = line.replace('\n', '')
                if line.startswith('S'):# TisaiYu[2024/11/13] 功能描述
                    parent_function, children_dot_split = line.split(":=")
                    if children_dot_split[0] == 'S':# TisaiYu[2024/11/13] 功能树描述
                        function_tree_description_lines.append(line)
                    else:
                        function_assembly_description_lines.append(line)
                elif line.startswith('g'):
                    assembly_part_description_lines.append(line)
                else:
                    continue
    function_node_edge_list = []
    part_id_list = []
    function_list = []
    function__record_list = defaultdict(list)
    assembly2part = {}
    function2part = {"ID":part_id_list,"function":function_list}
    for line in function_tree_description_lines:
        parent_function, children_dot_split = line.split(":=")
        children_node = children_dot_split.split(',')
        for child_node in children_node:
            function_node_edge_list.append((parent_function,child_node))
    for line in assembly_part_description_lines:
        assemgbly, part_dot_list = line.split(":=")
        part_list = part_dot_list.split(',')
        assembly2part[assemgbly] = part_list
    for line in function_assembly_description_lines:
        function, assembly_or_part_dot_list = line.split(":=")
        assembly_or_part_list = assembly_or_part_dot_list.split(',')
        for assembly_or_part in assembly_or_part_list:
            type_id = assembly_or_part[0]
            if type_id == 'g':
                for part_id in assembly2part[assembly_or_part]:
                    if part_id not in function2part['ID']:
                        function2part['ID'].append(part_id-1)
                    function__record_list[part_id].append(function)
            else:
                if assembly_or_part not in function2part['ID']:
                    function2part['ID'].append(assembly_or_part-1)
                function__record_list[assembly_or_part].append(function)
    function2part['ID'] = sorted(function2part['ID'],key=lambda x:int(x))
    for part_id in function2part['ID']:
        functions = function__record_list[part_id]
        functions_dot = ','.join(functions)
        function_list.append(functions_dot)
    df = pd.DataFrame(function2part)
    df["ID"] = df["ID"].astype(int)
    df = df.sort_values(by="ID")

    df["function"] = pd.Series(function_list)
    df.to_excel('SqlLoadExcel/function.xlsx',index=False)





if __name__ == "__main__":
    system_function_parse('11.11')