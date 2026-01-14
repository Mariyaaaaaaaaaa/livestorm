# # -*- encoding: utf-8 -*-
# '''
# @File    :   train.py
# @Time    :   2021/02/20 10:25:13
# @Author  :   Fei gao
# @Contact :   feig@mail.bnu.edu.cn
# BNU, Beijing, China
# '''
#
# # 导入必要的库
# import itertools
# import os
# import pandas as pd
# import scipy.sparse as sp
# from torch_geometric.data import Data
# import torch_geometric as tg
# import logging
# import argparse
# import networkx as nx
# import numpy as np
# import scipy
# import torch
# torch.autograd.set_detect_anomaly(True)
# from torch.utils.data import DataLoader
# import os
# from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
# from utils.minibatch import  MyDataset
# from utils.utilities import to_device
# from eval.link_prediction import evaluate_classifier
# from models.model import DySAT
#
# import torch
# torch.autograd.set_detect_anomaly(True)
#
#
# # dysat/100 : model.pt对应的最佳节点嵌入特征，batch_size=512
# # dysat/100_new_128 :model_new_128.pt对应的最佳节点嵌入特征，batch_size=128
# # dysat/100_new_256 :model_new_256.pt对应的最佳节点嵌入特征，batch_size=256
# # dysat/100_new_512 :model_new_512.pt对应的最佳节点嵌入特征，batch_size=512
#
#
# # 加载配置和数据
# def load_data(args):
#     graphs, adjs = load_graphs(args.dataset, args.num)
#     if args.featureless:
#         feats = [scipy.sparse.identity(args.all_nodes).tocsr()[range(0, x.shape[0]), :] for x in adjs if
#                  x.shape[0] <= args.all_nodes]
#     else:
#         # 如不是无特征数据，请提供替代加载特征的方法
#         pass
#     return graphs, adjs, feats
#
# def inductive_graph(graph_former, graph_later):
#     """Create the adj_train so that it includes nodes from (t+1)
#        but only edges from t: this is for the purpose of inductive testing.
#
#     Args:
#         graph_former ([type]): [description]
#         graph_later ([type]): [description]
#     """
#     newG = nx.MultiGraph()
#     newG.add_nodes_from(graph_later.nodes(data=True))
#     newG.add_edges_from(graph_former.edges(data=False))
#     return newG
#
# def _normalize_graph_gcn(adj):
#     """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
#     adj = sp.coo_matrix(adj, dtype=np.float32)
#     adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
#     rowsum = np.array(adj_.sum(1), dtype=np.float32)
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     return adj_normalized
#
# def _preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     features = np.array(features.todense())
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return features
#
#
# def build_pyg_graphs(features, adjs):
#     features = [_preprocess_features(feat) for feat in features]
#     adjs = [_normalize_graph_gcn(a) for a in adjs]
#     pyg_graphs = []
#     for feat, adj in zip(features, adjs):
#         x = torch.Tensor(feat)
#         edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
#         data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
#         pyg_graphs.append(data)
#     return pyg_graphs
#
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--time_steps', type=int, nargs='?', default=36,
#                         help="total time steps used for train, eval and test")
#     parser.add_argument('--all_nodes', type=int, nargs='?', default=508,
#                         help="total nodes in graphs")
#     parser.add_argument('--window_size', type=int, nargs='?', default=5,
#                         help="windows for train")
#     parser.add_argument('--stride', type=int, nargs='?', default=1,
#                         help="stride for train")
#     # Experimental settings.
#     parser.add_argument('--dataset', type=str, nargs='?', default='网络数据',
#                         help='dataset name')
#     parser.add_argument('--num', type=str, nargs='?', default=3,
#                         help='network name')
#     parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
#                         help='GPU_ID (0/1 etc.)')
#     parser.add_argument('--epochs', type=int, nargs='?', default=200,
#                         help='# epochs')
#     parser.add_argument('--val_freq', type=int, nargs='?', default=1,
#                         help='Validation frequency (in epochs)')
#     parser.add_argument('--test_freq', type=int, nargs='?', default=1,
#                         help='Testing frequency (in epochs)')
#     parser.add_argument('--batch_size', type=int, nargs='?', default=1024,
#                         help='Batch size (# nodes)')
#     parser.add_argument('--featureless', type=bool, nargs='?', default=True,
#                         help='True if one-hot encoding.')
#     parser.add_argument("--early_stop", type=int, default=10,
#                         help="patient")
#     # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
#     # Tunable hyper-params
#     # TODO: Implementation has not been verified, performance may not be good.
#     parser.add_argument('--residual', type=bool, nargs='?', default=True,
#                         help='Use residual')
#     # Number of negative samples per positive pair.
#     parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
#                         help='# negative samples per positive')
#     # Walk length for random walk sampling.
#     parser.add_argument('--walk_len', type=int, nargs='?', default=20,
#                         help='Walk length for random walk sampling')
#     # Weight for negative samples in the binary cross-entropy loss function.
#     parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
#                         help='Weightage for negative samples')
#     parser.add_argument('--learning_rate', type=float, nargs='?', default=0.001,
#                         help='Initial learning rate for self-attention model.')
#     parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
#                         help='Spatial (structural) attention Dropout (1 - keep probability).')
#     parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
#                         help='Temporal attention Dropout (1 - keep probability).')
#     parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
#                         help='Initial learning rate for self-attention model.')
#     # Architecture params
#     parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
#                         help='Encoder layer config: # attention heads in each GAT layer')
#     parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
#                         help='Encoder layer config: # units in each GAT layer')
#     parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
#                         help='Encoder layer config: # attention heads in each Temporal layer')
#     parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
#                         help='Encoder layer config: # units in each Temporal layer')
#     parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
#                         help='Position wise feedforward')
#     parser.add_argument('--window', type=int, nargs='?', default=-1,
#                         help='Window for temporal attention (default : -1 => full)')
#     args = parser.parse_args()
#
#
#     # 重置随机种子
#     np.random.seed(50)
#     # 加载数据
#     graphs, adjs, feats = load_data(args)
#
#     model_checkpoint_dir = "./model_checkpoints_l1"  # 模型保存目录
#
#     for start_idx in range(0, args.time_steps - args.window_size + 1, args.stride):
#         end_idx = start_idx + args.window_size
#         # 确保 end_idx 不超出列表范围
#         if end_idx > len(graphs):
#
#             end_idx = len(graphs)
#
#         # 创建用于保存模型的目录，以窗口大小命名
#         model_dir_name = f"window_{start_idx}_{end_idx - 1}"
#         model_dir_path = os.path.join(model_checkpoint_dir, model_dir_name)
#         os.makedirs(model_dir_path, exist_ok=True)
#
#         # 选择当前窗口内的图数据
#         window_graphs = graphs[start_idx:end_idx]
#         window_adjs = adjs[start_idx:end_idx]
#         window_feats = feats[start_idx:end_idx]
#
#         # 创建滑动窗口内的训练数据集和评估数据集
#         context_pairs_train = get_context_pairs(window_graphs, window_adjs)
#
#         # # inductive testing.
#         # new_G = inductive_graph(window_graphs[args.window_size - 2], window_graphs[args.window_size - 1])
#         # window_graphs[args.window_size - 1] = new_G
#         # window_adjs[args.window_size - 1] = nx.adjacency_matrix(new_G)
#
#         # 创建数据加载器和模型
#         dataset = MyDataset(args, window_graphs, window_feats, window_adjs, context_pairs_train)
#         dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,
#                                 collate_fn=MyDataset.collate_fn)
#
#         device = torch.device("cuda:0")
#         model = DySAT(args, window_feats[0].shape[1], args.window_size).to(device)
#         model_save_path = os.path.join(model_dir_path, "model.pt")
#
#         for idx, feed_dict in enumerate(dataloader):
#             feed_dict = to_device(feed_dict, device)
#
#         # 评估当前窗口的模型
#         model.load_state_dict(torch.load(model_save_path))
#         model.eval()
#         emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()
#
#         graph = window_graphs[-2]
#
#         # 获取图中实际存在的节点数
#         num_nodes = graph.number_of_nodes()
#         embeddings = emb[:num_nodes]  # 筛选出存在的节点的嵌入
#
#         # 提取节点和类别
#         nodes = list(graph.nodes)
#         categories = [graph.nodes[node]['category'] for node in nodes]
#
#         # 创建一个 DataFrame 保存节点、嵌入特征和类别
#         data = pd.DataFrame(embeddings, index=nodes)
#         data['Category'] = categories
#         data.reset_index(inplace=True)
#         data.rename(columns={'index': 'Node'}, inplace=True)
#
#         # 保存到 CSV 文件
#         file_path = f'./embedding/l1/embeddings_snapshot_{start_idx}.csv'
#         data.to_csv(file_path,
#                     header=['Node'] + [f'Embedding_{i + 1}' for i in range(embeddings.shape[1])] + ['Category'],
#                     index=False)




# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/02/20 10:25:13
@Author  :   Fei gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
# 导入必要的库
import pandas as pd
import os
import scipy.sparse as sp
from torch_geometric.data import Data
import torch_geometric as tg
import argparse
import networkx as nx
import numpy as np
import scipy
import torch
torch.autograd.set_detect_anomaly(True)
from utils.preprocess import load_graphs
from models.model import DySAT


# 加载配置和数据
def load_data(args):
    graphs, adjs = load_graphs(args.dataset, args.num)
    if args.featureless:
        feats = [scipy.sparse.identity(args.all_nodes).tocsr()[range(0, x.shape[0]), :] for x in adjs if
                 x.shape[0] <= args.all_nodes]
    else:
        # 如不是无特征数据，请提供替代加载特征的方法
        pass
    return graphs, adjs, feats

def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1)
       but only edges from t: this is for the purpose of inductive testing.

    Args:
        graph_former ([type]): [description]
        graph_later ([type]): [description]
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG

def _normalize_graph_gcn(adj):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj_.sum(1), dtype=np.float32)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    features = np.array(features.todense())
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def build_pyg_graphs(features, adjs):
    features = [_preprocess_features(feat) for feat in features]
    adjs = [_normalize_graph_gcn(a) for a in adjs]
    pyg_graphs = []
    for feat, adj in zip(features, adjs):
        x = torch.Tensor(feat)
        edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pyg_graphs.append(data)
    return pyg_graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=33,
                        help="total time steps used for train, eval and test")
    parser.add_argument('--all_nodes', type=int, nargs='?', default=3655,

                        help="total nodes in graphs")
    parser.add_argument('--window_size', type=int, nargs='?', default=5,
                        help="windows for train")
    parser.add_argument('--stride', type=int, nargs='?', default=1,
                        help="stride for train")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='网络数据',
                        help='dataset name')
    parser.add_argument('--num', type=str, nargs='?', default=87,
                        help='network name')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=1024,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    # Walk length for random walk sampling.
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.001,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each GAT layer')
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')
    args = parser.parse_args()
    print(args)

    # 重置随机种子
    np.random.seed(50)
    # 加载数据
    graphs, adjs, feats = load_data(args)

    model_checkpoint_dir = "./model_checkpoints_87"  # 模型保存目录

    # 外层循环遍历窗口
    for start_idx in range(0, args.time_steps - args.window_size + 1, args.stride):
        end_idx = start_idx + args.window_size
        if end_idx > len(graphs):
            end_idx = len(graphs)

        # 选择当前窗口内的图数据
        window_graphs = graphs[start_idx:end_idx]
        window_adjs = adjs[start_idx:end_idx]
        window_feats = feats[start_idx:end_idx]

        device = torch.device("cuda:0")
        pyg_graphs = build_pyg_graphs(window_feats, window_adjs)
        pyg_graphs = [graph.to(device) for graph in pyg_graphs]

        model_dir_name = f"window_{start_idx}_{end_idx - 1}"
        model_dir_path = os.path.join(model_checkpoint_dir, model_dir_name)
        model_save_path = os.path.join(model_dir_path, "model.pt")
        # 加载模型
        model = DySAT(args, window_feats[0].shape[1], args.window_size).to(device)
        model.load_state_dict(torch.load(os.path.join(model_save_path)))
        model.eval()

        node_embeddings = model(pyg_graphs)[:, -2, :].detach().cpu().numpy()

        # 创建输出目录（如果不存在）
        output_dir = "./embedding/87"
        os.makedirs(output_dir, exist_ok=True)

        # 保存node_embeddings到CSV文件
        output_file_path = os.path.join(output_dir, f"node_embeddings_{end_idx - 2}.csv")
        pd.DataFrame(node_embeddings).to_csv(output_file_path, index=False)








