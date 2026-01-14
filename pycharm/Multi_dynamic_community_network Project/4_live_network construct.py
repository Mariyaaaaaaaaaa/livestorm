
import networkx as nx
import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix
import pickle as pkl
from tqdm import tqdm
import concurrent.futures
import copy


# graph_snapshots_l0.pkl 初始总节点为N=100
def initialize_network(N0, m, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    G = nx.Graph()
    community_sizes = np.random.multinomial(N0, [1/m] * m)
    communities = []
    start = 0
    for size in community_sizes:
        community = list(range(start, start + size))
        G.add_nodes_from(community)
        G.add_edges_from([(community[i], community[j]) for i in range(len(community)) for j in range(i+1, len(community))])
        communities.append(community)
        start += size

    # 给每个社区的节点添加 category 标签，0是离线社区，1是在线社区
    for i, community in enumerate(communities):
        # 设置每个节点的 category 属性
        nx.set_node_attributes(G, {node: {'category': i} for node in community})

    for i in range(m):
        for j in range(i+1, m):
            u = random.choice(communities[i])
            v = random.choice(communities[j])
            G.add_edge(u, v)

    return G, communities

def redistribute_nodes_in_online_community(G, community, p, community_probs, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 度数小的优先转换
    # 计算社区中每个节点的度数
    community_degrees = {node: G.degree(node) for node in community}
    # 按照度数从小到大排序
    sorted_community_nodes = sorted(community_degrees, key=community_degrees.get)


    # 选择 p 倍的社区节点数量
    num_selected_nodes = int(p * len(community))
    # 将选中的一半节点加入相对社区，另一半节点保留在当前社区
    half = int(num_selected_nodes // 2)

    # 剩余的节点（1-p 倍）偏好加入某个社区
    remaining_nodes = len(community) - num_selected_nodes
    pre = int(remaining_nodes * community_probs[0])

    transferred_nodes = sorted_community_nodes[:half+pre]

    return transferred_nodes

def redistribute_community_nodes(G, communities, p, community_probs, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    communities_change = communities.copy()

    transferred_nodes_offline = redistribute_nodes_in_online_community(G, communities[1], p, community_probs, seed=None)

    # 更新社区和转移的节点
    communities_change[0] = communities[0] + transferred_nodes_offline
    communities_change[1] = [node for node in communities[1] if node not in transferred_nodes_offline]

    # 更新节点的 category 属性
    for i, community in enumerate(communities_change):
        nx.set_node_attributes(G, {node: {'category': i} for node in community})

    return G, communities_change, transferred_nodes_offline


def add_new_nodes(G, communities_change, Nt, p, community_probs, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    communities_new = communities_change.copy()

    total_nodes = len(G.nodes)
    new_nodes = list(range(total_nodes, total_nodes + Nt))
    G.add_nodes_from(new_nodes)

    # 打乱新节点的顺序
    random.shuffle(new_nodes)

    # 选择 p 倍的新节点数量
    num_selected_nodes = int(p * len(new_nodes))
    selected_nodes = new_nodes[:num_selected_nodes]
    # 将选中的一半节点加入第一个社区，另一半加入第二个社区
    half = int(len(selected_nodes) // 2)
    communities_new[0].extend(selected_nodes[:half])
    communities_new[1].extend(selected_nodes[half:num_selected_nodes])

    # 剩余的节点（1-p 倍）偏好加入某个社区
    remaining_nodes = new_nodes[num_selected_nodes:]


    # 计算每个社区的加入观众数
    pre = int(len(remaining_nodes) * community_probs[0])
    communities_new[0].extend(remaining_nodes[:pre])
    communities_new[1].extend(remaining_nodes[pre:])

    # 给每个社区的节点添加 category 标签，0是离线社区，1是在线社区
    for i, community in enumerate(communities_new):
        # 设置每个节点的 category 属性
        nx.set_node_attributes(G, {node: {'category': i} for node in community})

    return G, communities_new, new_nodes


def remove_and_add_edges(G, transferred_nodes, m_edges_0, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    for node in transferred_nodes:
        # 移除节点的所有边
        G.remove_edges_from(list(G.edges(node)))

    for node in transferred_nodes:
        # 获取节点的邻居节点
        neighbors = set(G.neighbors(node))
        # 获取社区中除了当前节点的其他节点，并排除邻居节点
        community_nodes = [n for n in transferred_nodes if n != node and n not in neighbors]

        actual_m_edges = min(m_edges_0, len(community_nodes))
        # 基于节点度数选择目标节点
        community_targets = random.sample(community_nodes,k=actual_m_edges)
        # 添加新边
        G.add_edges_from([(node, target) for target in community_targets])

    return G


def add_new_edges(G, communities_new, new_nodes, m_edges_0, m_edges_1, n_edges, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    for node in new_nodes:
        # 找到节点所属的社区
        node_category = G.nodes[node]['category']
        community = communities_new[node_category]
        other_community = communities_new[1-node_category]
        community_degrees = {n: G.degree(n) for n in community if n != node}
        other_degrees = {n: G.degree(n) for n in other_community}

        # 根据社区类别选择内部边数
        if G.nodes[node]['category'] == 0:
            actual_m_edges = min(m_edges_0, len(community))
        else:
            actual_m_edges = min(m_edges_1, len(community))

        actual_n_edges = min(n_edges, len(other_community))

        #采样
        community_targets = random.choices(list(community_degrees.keys()), weights=list(community_degrees.values()),
                                          k=actual_m_edges)
        other_targets = random.choices(list(other_degrees.keys()), weights=list(other_degrees.values()),
                                      k=actual_n_edges)

        G.add_edges_from([(node, target) for target in community_targets])
        G.add_edges_from([(node, target) for target in other_targets])



def dynamic_community_network(N0, m, Nt_change, P, Q, T, m_edges_0, m_edges_1, n_edges, seed=None):
    G, communities = initialize_network(N0, m, seed)
    snapshots = [G.copy()]
    N_t1 = N0

    for t in tqdm(range(0, T), desc="Processing snapshots"):
        Nt = int(N_t1 * (1 + Nt_change.iloc[t, 0]))
        p = P[t]
        q = Q[t]

        # 计算未更新前的社区大小
        original_community_sizes = [len(c) for c in communities]
        community_probs = [size ** q for size in original_community_sizes]
        if np.sum(community_probs) == 0:
            community_probs = [0.5, 0.5]
        else:
            community_probs = community_probs / np.sum(community_probs)

#需要将上一时刻的离线节点作为孤立节点
        # 重新分配社区节点并记录转移的节点
        G, communities_change, transferred_nodes = redistribute_community_nodes(G, communities, p, community_probs, seed)
        # 添加新节点并记录新增的节点
        G, communities_new, new_nodes = add_new_nodes(G, communities_change, Nt, p, community_probs, seed)

        for node in communities[0]:
            # 移除节点的所有边
            G.remove_edges_from(list(G.edges(node)))

        communities_new[0] = [node for node in communities_new[0] if node not in communities[0]]

        # 移除并添加边，针对转移的节点
        remove_and_add_edges(G, transferred_nodes, m_edges_0, seed)
        # 添加新边，针对新增的节点
        add_new_edges(G, communities_new, new_nodes, m_edges_0, m_edges_1, n_edges, seed)

        snapshots.append(G.copy())
        N_t1 = Nt
        communities = communities_new

    return snapshots

# Example usage
CHANGE_DATA_PATH = 'D:/研究生/毕业论文/代码/网络构建/直播平台网络模型/jupyter/处理数据/change_rate_real_time_user_level_1.csv'
BEST_PARAMS_PATH = './实验结果/params_and_distances_r=50_l1_test.csv'
Nt_change = pd.read_csv(CHANGE_DATA_PATH, index_col=0)
Best_params = pd.read_csv(BEST_PARAMS_PATH, index_col=0)
# 提取第一个参数 p 和第二个参数 q
P = Best_params['Best Parameters'].apply(lambda x: float(x.strip('()').split(',')[0]))
Q = Best_params['Best Parameters'].apply(lambda x: float(x.strip('()').split(',')[1]))
N = 100
m = 2
T = 35
m_edges_0 = 2  # 社区0的节点内部连边数
m_edges_1 = 3  # 社区1的节点内部连边数
n_edges = 1
seed = 50  # 设置随机数种子

# [5110,6137][0.45,0.55]
N0 = 55

snapshots = dynamic_community_network(N0, m, Nt_change, P, Q, T, m_edges_0, m_edges_1, n_edges, seed)

# 保存数据
save_path = "./网络数据/graph_snapshots_l1_test.pkl"
with open(save_path, "wb") as f:
    pkl.dump(snapshots, f)

print("Processed Data Saved at {}".format(save_path))

# def redistribute_community_nodes(G, communities, p, q, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#         random.seed(seed)
#
#     # 计算未更新前的社区大小
#     original_community_sizes = [len(c) for c in communities]
#
#     # 打乱社区1中的节点顺序
#     community1_nodes = communities[1].copy()
#     random.shuffle(community1_nodes)
#
#     # 选择 p 倍的社区1节点数量
#     num_selected_nodes = int(p * len(community1_nodes))
#     selected_nodes = community1_nodes[:num_selected_nodes]
#     # 将选中的一半节点加入社区0，另一半节点保留在社区1
#     half = int(len(selected_nodes) // 2)
#     transferred_nodes = selected_nodes[:half]
#     communities[0].extend(transferred_nodes)
#     communities[1] = [node for node in communities[1] if node not in transferred_nodes]
#
#     # 剩余的节点（1-p 倍）偏好加入某个社区
#     remaining_nodes = community1_nodes[num_selected_nodes:]
#     community_probs = [size ** q for size in original_community_sizes]
#     if np.sum(community_probs) == 0:
#         community_probs = [0.5, 0.5]
#     else:
#         community_probs = community_probs / np.sum(community_probs)
#
#     pre = int(len(remaining_nodes) * community_probs[0])
#     additional_transferred_nodes = remaining_nodes[:pre]
#     communities[0].extend(additional_transferred_nodes)
#     communities[1] = [node for node in communities[1] if node not in additional_transferred_nodes]
#
#     # 更新节点的 category 属性
#     for i, community in enumerate(communities):
#         nx.set_node_attributes(G, {node: {'category': i} for node in community})
#
#     # 将新转移的节点添加到 transferred_nodes
#     transferred_nodes.extend(additional_transferred_nodes)
#
#     return G, communities, transferred_nodes



## 只考虑在线节点的转换
# import networkx as nx
# import numpy as np
# import pandas as pd
# import random
# from scipy.sparse import csr_matrix
# import pickle as pkl
# from tqdm import tqdm
#
#
# def initialize_network(N0, m, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#         random.seed(seed)
#
#     G = nx.Graph()
#     community_sizes = np.random.multinomial(N0, [1/m] * m)
#     communities = []
#     start = 0
#     for size in community_sizes:
#         community = list(range(start, start + size))
#         G.add_nodes_from(community)
#         G.add_edges_from([(community[i], community[j]) for i in range(len(community)) for j in range(i+1, len(community))])
#         communities.append(community)
#         start += size
#
#     # 给每个社区的节点添加 category 标签，0是离线社区，1是在线社区
#     for i, community in enumerate(communities):
#         # 设置每个节点的 category 属性
#         nx.set_node_attributes(G, {node: {'category': i} for node in community})
#
#     for i in range(m):
#         for j in range(i+1, m):
#             u = random.choice(communities[i])
#             v = random.choice(communities[j])
#             G.add_edge(u, v)
#
#     return G, communities
#
# def redistribute_community_nodes(G, communities, p, q, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#         random.seed(seed)
#
#     # 计算未更新前的社区大小
#     original_community_sizes = [len(c) for c in communities]
#
#     # 打乱社区1中的节点顺序
#     community1_nodes = communities[1].copy()
#     random.shuffle(community1_nodes)
#
#     # 选择 p 倍的社区1节点数量
#     num_selected_nodes = int(p * len(community1_nodes))
#     selected_nodes = community1_nodes[:num_selected_nodes]
#     # 将选中的一半节点加入社区0，另一半节点保留在社区1
#     half = int(len(selected_nodes) // 2)
#     transferred_nodes = selected_nodes[:half]
#     communities[0].extend(transferred_nodes)
#     communities[1] = [node for node in communities[1] if node not in transferred_nodes]
#
#     # 剩余的节点（1-p 倍）偏好加入某个社区
#     remaining_nodes = community1_nodes[num_selected_nodes:]
#     community_probs = [size ** q for size in original_community_sizes]
#     if np.sum(community_probs) == 0:
#         community_probs = [0.5, 0.5]
#     else:
#         community_probs = community_probs / np.sum(community_probs)
#
#     pre = int(len(remaining_nodes) * community_probs[0])
#     additional_transferred_nodes = remaining_nodes[:pre]
#     communities[0].extend(additional_transferred_nodes)
#     communities[1] = [node for node in communities[1] if node not in additional_transferred_nodes]
#
#     # 更新节点的 category 属性
#     for i, community in enumerate(communities):
#         nx.set_node_attributes(G, {node: {'category': i} for node in community})
#
#     # 将新转移的节点添加到 transferred_nodes
#     transferred_nodes.extend(additional_transferred_nodes)
#
#     return G, communities, transferred_nodes
# def add_new_nodes(G, communities, Nt, p, q, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#         random.seed(seed)
#
#     total_nodes = len(G.nodes)
#     new_nodes = list(range(total_nodes, total_nodes + Nt))
#     G.add_nodes_from(new_nodes)
#
#     # 计算未更新前的社区大小
#     original_community_sizes = [len(c) for c in communities]
#
#     # 打乱新节点的顺序
#     random.shuffle(new_nodes)
#
#     # 选择 p 倍的新节点数量
#     num_selected_nodes = int(p * len(new_nodes))
#     selected_nodes = new_nodes[:num_selected_nodes]
#     # 将选中的一半节点加入第一个社区，另一半加入第二个社区
#     half = int(len(selected_nodes) // 2)
#     communities[0].extend(selected_nodes[:half])
#     communities[1].extend(selected_nodes[half:num_selected_nodes])
#
#     # 剩余的节点（1-p 倍）偏好加入某个社区
#     remaining_nodes = new_nodes[num_selected_nodes:]
#
#     community_probs = [size ** q for size in original_community_sizes]
#     if np.sum(community_probs) == 0:
#         community_probs = [0.5, 0.5]
#     else:
#         community_probs = community_probs / np.sum(community_probs)
#     # 计算每个社区的加入观众数
#     pre = int(len(remaining_nodes) * community_probs[0])
#     communities[0].extend(remaining_nodes[:pre])
#     communities[1].extend(remaining_nodes[pre:])
#
#     # 给每个社区的节点添加 category 标签，0是离线社区，1是在线社区
#     for i, community in enumerate(communities):
#         # 设置每个节点的 category 属性
#         nx.set_node_attributes(G, {node: {'category': i} for node in community})
#
#     return G, communities, new_nodes
#
# def remove_and_add_edges(G, transferred_nodes, communities, m_edges_0, m_edges_1, n_edges, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#         random.seed(seed)
#
#     for node in transferred_nodes:
#         # 移除节点的所有边
#         G.remove_edges_from(list(G.edges(node)))
#
#         # 添加新边
#         community = [c for c in communities if node in c][0]
#         community_degrees = {n: G.degree(n) for n in community if n != node}
#         other_degrees = {n: G.degree(n) for c in communities for n in c if n != node and n not in community}
#
#         # 根据社区类别选择内部边数
#         if G.nodes[node]['category'] == 0:
#             actual_m_edges = min(m_edges_0, len(community_degrees))
#         else:
#             actual_m_edges = min(m_edges_1, len(community_degrees))
#
#         actual_n_edges = min(n_edges, len(other_degrees))
#
#         community_targets = random.choices(list(community_degrees.keys()), weights=list(community_degrees.values()),
#                                            k=actual_m_edges)
#         other_targets = random.choices(list(other_degrees.keys()), weights=list(other_degrees.values()),
#                                        k=actual_n_edges)
#
#         G.add_edges_from([(node, target) for target in community_targets])
#         G.add_edges_from([(node, target) for target in other_targets])
#
# def add_new_edges(G, communities, new_nodes, m_edges_0, m_edges_1, n_edges, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#         random.seed(seed)
#
#     for node in new_nodes:
#         community = [c for c in communities if node in c][0]
#         community_degrees = {n: G.degree(n) for n in community if n != node}
#         other_degrees = {n: G.degree(n) for c in communities for n in c if n != node and n not in community}
#
#         # 根据社区类别选择内部边数
#         if G.nodes[node]['category'] == 0:
#             actual_m_edges = min(m_edges_0, len(community_degrees))
#         else:
#             actual_m_edges = min(m_edges_1, len(community_degrees))
#
#         actual_n_edges = min(n_edges, len(other_degrees))
#
#         community_targets = random.choices(list(community_degrees.keys()), weights=list(community_degrees.values()),
#                                            k=actual_m_edges)
#         other_targets = random.choices(list(other_degrees.keys()), weights=list(other_degrees.values()),
#                                        k=actual_n_edges)
#         G.add_edges_from([(node, target) for target in community_targets])
#         G.add_edges_from([(node, target) for target in other_targets])
#
# def dynamic_community_network(N0, m, Nt_change, P, Q, T, m_edges_0, m_edges_1, n_edges, seed=None):
#     G, communities = initialize_network(N0, m, seed)
#     snapshots = [G.copy()]
#     N_t1 = N0
#
#     for t in tqdm(range(1, T), desc="Processing snapshots"):
#         Nt = int(N_t1 * (1 + Nt_change.iloc[t, 0]))
#         p = P[t]
#         q = Q[t]
#
#         # 重新分配社区节点并记录转移的节点
#         G, communities, transferred_nodes = redistribute_community_nodes(G, communities, p, q, seed)
#
#         # 移除并添加边，针对转移的节点
#         remove_and_add_edges(G, transferred_nodes, communities, m_edges_0, m_edges_1, n_edges, seed)
#
#         # 添加新节点并记录新增的节点
#         G, communities, new_nodes = add_new_nodes(G, communities, Nt, p, q, seed)
#
#         # 添加新边，针对新增的节点
#         add_new_edges(G, communities, new_nodes, m_edges_0, m_edges_1, n_edges, seed)
#
#         snapshots.append(G.copy())
#         N_t1 = Nt
#
#     return snapshots
#
# # Example usage
# #使用变化率的平均值不太可行，网络会巨大，因此使用平均值的变化率。
# CHANGE_DATA_PATH = 'D:/lunwen/social_network2023.7.17/daima/zhibo/新实验/Jupyter code/Nt_change_34_avg.csv'
# BEST_PARAMS_PATH = 'D:/lunwen/social_network2023.7.17/daima/zhibo/新实验/Jupyter code/best_pq.csv'
# Nt_change = pd.read_csv(CHANGE_DATA_PATH, index_col=0)
# Best_params = pd.read_csv(BEST_PARAMS_PATH, index_col=0)
# P = Best_params.iloc[:, 0]
# Q = Best_params.iloc[:, 1]
# N0 = 300
# m = 2
# T = 33
# m_edges_0 = 2  # 社区0的节点内部连边数
# m_edges_1 = 3  # 社区1的节点内部连边数
# n_edges = 1
# seed = 50  # 设置随机数种子
#
# snapshots = dynamic_community_network(N0, m, Nt_change, P, Q, T, m_edges_0, m_edges_1, n_edges, seed)
#
# # One-hot 编码和保存图快照
# onehot = np.identity(len(snapshots[-1].nodes()), dtype=int)
# for snapshot in snapshots:
#     # 创建 One-hot 特征
#     onehot_features = []
#     for node in snapshot.nodes():
#         onehot_features.append(onehot[node])
#
#     # 将 One-hot 特征转换为稀疏矩阵
#     onehot_features_csr = csr_matrix(onehot_features)
#
#     snapshot.graph["feature"] = onehot_features_csr
#
# # 保存数据
# save_path = "./网络数据/graph_snapshots_300_new.pkl"
# with open(save_path, "wb") as f:
#     pkl.dump(snapshots, f)
#
# print("Processed Data Saved at {}".format(save_path))