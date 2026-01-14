# 导入必要的库
import itertools
import pandas as pd
import scipy.sparse as sp
import logging
import argparse
import networkx as nx
import numpy as np
import scipy
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import pickle
from scipy.special import expit  # scipy.special.expit 是 sigmoid 函数的实现

# log_win15.txt  729it [23:55,  1.97s/it]
# log_win15_76.txt  最优种子数  729it [11:09,  1.09it/s]

def load_snapshots(file_path):
    with open(file_path, 'rb') as f:
        snapshots = pickle.load(f)
    return snapshots

def calculate_embedding_product(embeddings):
    # 计算 embeddings 与其转置的矩阵乘法
    embedding_matrix = embeddings.values
    # 计算嵌入矩阵的乘积
    embedding_product_matrix = np.dot(embedding_matrix, embedding_matrix.T)
    # 将结果输入到 sigmoid 函数中
    sigmoid_matrix = expit(embedding_product_matrix)
    # 从 sigmoid 结果中减去 1
    final_matrix = 1 - sigmoid_matrix
    # 保留原始的索引
    return pd.DataFrame(final_matrix)


def calculate_similarity_matrix(node_embeddings, node_states, state_diff_weight):
    # 将 node_states 字典转换为带有标签的 NumPy 数组，保留 node_embeddings 的列标签顺序
    node_state_series = pd.Series(node_states)
    node_state_values = node_state_series.values
    # 计算状态差异矩阵
    state_diff_matrix = np.abs(node_state_values[:, np.newaxis] - node_state_values[np.newaxis, :])
    state_diff_matrix_df = pd.DataFrame(state_diff_matrix, index=node_embeddings.columns,
                                        columns=node_embeddings.columns)

    # 合并嵌入相似度和状态差异
    similarity_matrix = (1 - state_diff_weight) * node_embeddings + state_diff_weight * state_diff_matrix_df

    return similarity_matrix


def find_neighbors_bool_matrix(active_nodes, node_embeddings, opinions, state_diff_weight, similarity_threshold):
    active_nodes = list(active_nodes)

    # 提取活跃节点的嵌入
    active_embeddings = node_embeddings.loc[active_nodes, active_nodes]

    # 计算相似性矩阵
    similarity_matrix = calculate_similarity_matrix(active_embeddings, opinions, state_diff_weight)

    # 创建一个布尔矩阵来表示是否是邻居
    bool_matrix = similarity_matrix <= similarity_threshold
    # 将布尔矩阵转换为 NumPy 数组
    bool_matrix_np = bool_matrix.to_numpy()
    np.fill_diagonal(bool_matrix_np, False)  # 自己不作为自己的邻居

    return bool_matrix

# 直播网络

# 设置随机种子
np.random.seed(1)

# 设置参数网格
similarity_threshold_values = np.arange(0.1, 1, step=0.1).round(1)
state_diff_weight_values = np.arange(0.1, 1, step=0.1).round(1)
update_speed_values = np.arange(0.1, 1, step=0.1).round(1)

num_iterations = 180  # 迭代次数

log_file = "./logs/log_win15_l1.txt"  # 设置日志文件名
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# 加载数据
dataset_path = './网络数据/graph_snapshots_l1.pkl'
graphs = load_snapshots(dataset_path)

# 外层循环遍历窗口
start_idx = 14
end_idx = start_idx + 5
if end_idx > len(graphs):
    end_idx = len(graphs)

# 选择当前窗口内的图数据
window_graphs = graphs[start_idx:end_idx]

# 读取CSV文件
file_path = f'./node_embeddings/l1/node_embeddings_{end_idx-2}.csv'
node_embeddings = pd.read_csv(file_path)
final_embeddings = calculate_embedding_product(node_embeddings)

active_nodes = [node for i, node in enumerate(window_graphs[3].nodes()) if
                window_graphs[3].nodes[node]['category'] == 1]
# 初始化opinions
opinions_0 = {node: np.random.uniform(0, 1) for node in active_nodes}

# 内层循环遍历参数组合
for parameters in tqdm(
        itertools.product(similarity_threshold_values, state_diff_weight_values, update_speed_values)):
    similarity_threshold, state_diff_weight, update_speed = parameters
    # 记录初始观点
    logging.info(f"Parameters: {parameters}, Iteration: 0, Initial Opinions: {opinions_0}")
    opinions = opinions_0.copy()
    opinions_over_time = []
    no_change_count = 0

    for iteration in range(num_iterations):
        opinions_over_time.append(opinions.copy())
        bool_matrix = find_neighbors_bool_matrix(active_nodes, final_embeddings, opinions,state_diff_weight, similarity_threshold)
        for i in active_nodes:
            neighbor_indices = np.where(bool_matrix[i])[0]
            if len(neighbor_indices) > 0:
                neighbor_labels = np.array(active_nodes)[neighbor_indices]
                neighbor_opinions_values = np.array([opinions[node] for node in neighbor_labels])
                neighbor_opinions_mean = np.mean(neighbor_opinions_values)
                opinions[i] += update_speed * (neighbor_opinions_mean - opinions[i])
        # 记录最终观点
        logging.info(f"Parameters: {parameters}, Iteration: {iteration+1}, Opinions: {opinions}")

        if len(opinions_over_time) > 1:
            current_opinions = np.array(list(opinions.values()))
            previous_opinions = np.array(list(opinions_over_time[-1].values()))

            diff = current_opinions - previous_opinions  # 计算差值

            # 设置一个差值阈值，例如 0.001（根据你的需求调整）
            threshold = 0.00001

            if np.all(np.abs(diff) < threshold):
                no_change_count += 1
            else:
                no_change_count = 0

        if no_change_count == 5:
            break

