# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/02/20 10:25:13
@Author  :   Fei gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import argparse
import networkx as nx
import numpy as np
import dill
import pickle as pkl
import scipy
from torch.utils.data import DataLoader
import os

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from models.model import DySAT

import torch
torch.autograd.set_detect_anomaly(True)



from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary


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

    graphs, adjs = load_graphs(args.dataset, args.num)
    if args.featureless == True:
        feats = [scipy.sparse.identity(args.all_nodes).tocsr()[range(0, x.shape[0]), :] for x in adjs if
                 x.shape[0] <= args.all_nodes]

    assert args.time_steps <= len(adjs), "Time steps is illegal"

    best_auc_test = 0.0  # 用于跟踪最佳测试AUC的变量
    early_stop = 5  # 早停次数
    model_checkpoint_dir = "./model_checkpoints_87"  # 模型保存目录

    for start_idx in range(0, args.time_steps - args.window_size + 1, args.stride):
        end_idx = start_idx + args.window_size
        # 确保 end_idx 不超出列表范围
        if end_idx > len(graphs):

            end_idx = len(graphs)

        # 创建用于保存模型的目录，以窗口大小命名
        model_dir_name = f"window_{start_idx}_{end_idx - 1}"
        model_dir_path = os.path.join(model_checkpoint_dir, model_dir_name)
        os.makedirs(model_dir_path, exist_ok=True)

        # 选择当前窗口内的图数据
        window_graphs = graphs[start_idx:end_idx]
        window_adjs = adjs[start_idx:end_idx]
        window_feats = feats[start_idx:end_idx]

        # 创建滑动窗口内的训练数据集和评估数据集
        context_pairs_train = get_context_pairs(window_graphs, window_adjs)
        train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg = get_evaluation_data(
            window_graphs)

        # inductive testing.
        new_G = inductive_graph(window_graphs[args.window_size - 2], window_graphs[args.window_size - 1])
        window_graphs[args.window_size - 1] = new_G
        window_adjs[args.window_size - 1] = nx.adjacency_matrix(new_G)

        # 创建数据加载器和模型
        dataset = MyDataset(args, window_graphs, window_feats, window_adjs, context_pairs_train)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,
                                collate_fn=MyDataset.collate_fn)

        device = torch.device("cuda:0")
        model = DySAT(args, window_feats[0].shape[1], args.window_size).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_epoch_val = 0
        patient = 0

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = []
            for idx, feed_dict in enumerate(dataloader):
                feed_dict = to_device(feed_dict, device)
                opt.zero_grad()
                loss = model.get_loss(feed_dict)
                loss.backward()
                opt.step()
                epoch_loss.append(loss.item())

            model.eval()
            emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()
            val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                                  train_edges_neg,
                                                                  val_edges_pos,
                                                                  val_edges_neg,
                                                                  test_edges_pos,
                                                                  test_edges_neg,
                                                                  emb,
                                                                  emb)
            epoch_auc_val = val_results["HAD"][1]
            epoch_auc_test = test_results["HAD"][1]

            if epoch_auc_val > best_epoch_val:
                best_epoch_val = epoch_auc_val
                model_save_path = os.path.join(model_dir_path, "model.pt")
                torch.save(model.state_dict(), model_save_path)
                patient = 0
            else:
                patient += 1
                if patient > early_stop:
                    break

            print("Window [{:<2}:{:<2}] Epoch {:<3}, Loss = {:.3f}, Val AUC {:.3f} Test AUC {:.3f}".format(
                start_idx, end_idx - 1, epoch, np.mean(epoch_loss), epoch_auc_val, epoch_auc_test))

        # 评估当前窗口的模型
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()
        val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                              train_edges_neg,
                                                              val_edges_pos,
                                                              val_edges_neg,
                                                              test_edges_pos,
                                                              test_edges_neg,
                                                              emb,
                                                              emb)
        auc_test = test_results["HAD"][1]
        print("Window [{:<2}:{:<2}] Best Test AUC = {:.3f}".format(start_idx, end_idx - 1, auc_test))

        if auc_test > best_auc_test:
            best_auc_test = auc_test

    print("Overall Best Test AUC = {:.3f}".format(best_auc_test))






