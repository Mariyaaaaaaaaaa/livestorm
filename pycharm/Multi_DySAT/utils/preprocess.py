
import numpy as np
import dill
import pickle
import networkx as nx
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from utils.utilities import run_random_walks_n2v

np.random.seed(50)

# def load_graphs(dataset_str):
#     """Load graph snapshots given the name of dataset"""
#     with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
#         graphs = pkl.load(f)
#     print("Loaded {} graphs ".format(len(graphs)))
#     adjs = [nx.adjacency_matrix(g) for g in graphs]
#     return graphs, adjs


# def load_graphs(dataset_str, num_time_steps):
#     graphs = []
#
#     for time_step in range(num_time_steps):
#         try:
#             load_path = f"data/{dataset_str}/graph_snapshot_{time_step}.pkl"
#             with open(load_path, "rb") as f:
#                 G = pkl.load(f)
#                 graphs.append(G)
#         except Exception as e:
#             print(f"Error while loading data for time step {time_step}: {str(e)}")
#
#     print(f"Loaded {len(graphs)} graphs")
#     adjs = [nx.adjacency_matrix(g) for g in graphs]
#     return graphs, adjs

def load_graphs(dataset_str, num):
    file_path = f"data/{dataset_str}/graph_snapshots_{num}_true.pkl"
    with open(file_path, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs")
    adjs = [nx.adjacency_matrix(g) for g in graphs]

    return graphs, adjs



def get_context_pairs(graphs, adjs):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    print("Computing training pairs ...")
    context_pairs_train = []
    for i in range(len(graphs)):
        context_pairs_train.append(run_random_walks_n2v(graphs[i], adjs[i], num_walks=10, walk_len=20))
    return context_pairs_train

def get_evaluation_data(graphs):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx+1]
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2, 
                            test_mask_fraction=0.6)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):
            edges_positive.append(e)
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)


    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive,
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos,
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg


def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg