import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def calculate_pos(self, G, classified_labels):
    # 获取所有节点
    nodes = list(G.nodes())

    # 获取唯一的类别
    unique_labels = set(classified_labels)

    # 为每个类别分配一个中心位置
    label_positions = {}
    for i, label in enumerate(unique_labels):
        # 为每个类别生成一个随机中心位置
        label_positions[label] = np.array([i * 2, 0])  # 这里可以调整类别之间的间距

    # 计算每个节点的位置
    pos = {}
    for node in nodes:
        label = classified_labels[node]
        # 将节点放置在其类别的中心附近
        pos[node] = label_positions[label] + np.random.normal(0, 0.1, 2)  # 添加一些随机噪声

    return pos


# 示例使用
G = nx.Graph()
# 添加节点和边
G.add_nodes_from(range(10))
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (8, 9)])

# 假设每个节点的类别
classified_labels = {0: 'A', 1: 'A', 2: 'B', 3: 'B', 4: 'C', 5: 'A', 6: 'C', 7: 'C', 8: 'B', 9: 'A'}

# 计算节点位置
pos = calculate_pos(None, G, classified_labels)

# 绘制图形
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color=[classified_labels[node] for node in G.nodes()],
        node_size=500, font_size=10, font_color='white', edge_color='gray')
plt.title("Graph with Nodes Positioned by Class")
plt.show()