import torch.nn.init as init
import os.path as osp
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.data import Data
import torch_geometric.transforms as T
from tqdm import tqdm
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device used for computation
# train num of datasets split in training sets
# epsilon = 0.05
# train_num = 40  # single class
# test_num = 1000
# dataset = 'BA'  # dataset
# # BA parameter
# base_num = 1000
# path_num = 5
# house_num = 200
# random_num = 10
# test_num = 1000
# base_num = 1000
# path_num = 5
# house_num = 200
# random_num = 10
# feature_dim = 256
def edge2graph(edge_index, node_num):
    graph = torch.zeros((node_num, node_num))
    for i in range(edge_index.shape[1]):
        graph[edge_index[0][i]][edge_index[1][i]] = 1
        # graph[edge_index[1][i], edge_index[0][i]] = 1
    return graph


def uniform_sampling(node_num, class_num, label, train_num, test_num):
    node_list = torch.tensor(range(node_num))
    train_list = []
    for i in range(class_num):
        # print(i)
        candidate = [label == i]
        index = list(node_list[candidate])
        sample = list(random.sample(index, train_num))
        train_list.extend(sample)
        # print(len(train_list))
    train_list.sort()
    test_candidate = list(set(node_list) - set(train_list))
    test_candidate.sort()
    test_list = list(random.sample(list(test_candidate), test_num))
    test_list.sort()
    # print(test_list)
    tensor_train_mask = [False] * node_num
    tensor_test_mask = [False] * node_num
    for i in range(train_num * class_num):
        tensor_train_mask[train_list[i]] = True
    for i in range(test_num):
        tensor_test_mask[test_list[i]] = True
    tensor_train_mask = torch.tensor(tensor_train_mask)
    tensor_test_mask = torch.tensor(tensor_test_mask)
    return tensor_train_mask, tensor_test_mask


def random_sampling(node_num, class_num, train_num, test_num):
    train_num = train_num * class_num
    list1 = torch.tensor(np.arange(node_num - 1))
    train_list = list(random.sample(list(list1), train_num + test_num))
    tensor_train_mask = [False] * node_num
    tensor_test_mask = [False] * node_num
    for i in range(train_num):
        tensor_train_mask[train_list[i]] = True
    for i in range(test_num):
        tensor_test_mask[train_list[i + train_num]] = True
    tensor_train_mask = torch.tensor(tensor_train_mask)
    tensor_test_mask = torch.tensor(tensor_test_mask)
    return tensor_train_mask, tensor_test_mask


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'dblp']
    if name == 'dblp':
        return CitationFull(
            path,
            name,
            transform=T.NormalizeFeatures())
    else:
        return Planetoid(
            path,
            name,
            transform=T.NormalizeFeatures())


def politic_blog(feature_dim):
    f1 = open("datasets/PolBlogs/raw/adjacency.txt")
    f2 = open("datasets/PolBlogs/raw/label.txt")
    edge_list = []
    label_list = []
    for line in f1.readlines():
        line = line.split('\t')
        edge_list.append([int(line[0]), int(line[1])])
        edge_list.append([int(line[1]), int(line[0])])
    edge_index = torch.tensor(edge_list).t().contiguous()
    for line2 in f2.readlines():
        line2 = line2.strip('\n')
        label_list.append(eval(line2))
    label_list = torch.tensor(label_list).view(-1)
    x = torch.rand(label_list.shape[0], feature_dim)
    # x = torch.zeros(label_list.shape[0], 128)
    data = Data(edge_index=edge_index, x=x, y=label_list, num_nodes=label_list.size(0))
    f1.close()
    f2.close()
    return data


# BA-Shape
def BA_Shape(n, m, house_num, random_num):
    graph1 = nx.barabasi_albert_graph(n, m)
    label_list = []
    for i in range(n):
        label_list.append(0)

    for i in range(house_num):
        start = n + (i * 5)
        graph1.add_nodes_from(range(start, start + 5))
        graph1.add_edges_from(
            [
                (start, start + 1),
                (start + 1, start + 2),
                (start + 2, start + 3),
                (start + 3, start + 4),
                (start + 4, start)
            ]
        )
        list = [1, 1, 1, 1, 1]
        label_list.extend(list)
        v1 = random.sample(range(5), 1)
        v2 = random.sample(range(n), 1)
        graph1.add_edges_from([(start + v1[0], v2[0])])  # 连接
    # 扰动
    for i in range(random_num):
        e = random.sample(range(n + 5 * house_num), 2)
        graph1.add_edges_from([(e[0], e[1])])
    # adj = np.array(nx.adjacency_matrix(graph1).todense())
    return graph1, label_list


def BA(base_num, path_num, house_num, random_num):
    graph, label = BA_Shape(n=base_num, m=path_num, house_num=house_num, random_num=random_num)
    edge_list = []
    for edge in graph.edges():
        edge_list.append([edge[0], edge[1]])
        edge_list.append([edge[1], edge[0]])
    edge_index = torch.tensor(edge_list).t().contiguous()
    label_list = torch.tensor(label).view(-1)
    x = torch.rand(label_list.shape[0], 256)
    # x = torch.ones(label_list.shape[0], 256)
    data = Data(edge_index=edge_index, x=x, y=label_list, num_nodes=label_list.size(0))
    return data


def build_data(data, node_num, device, epsilon):
    """
    :param dataset: Cora, CiteSeer, PubMed or PolBlog
    :param data: data
    :return: graph adjacency matrix, node feature and label
    """
    # build adjacency
    Adj = edge2graph(data.edge_index, node_num)
    edge_epsilon = data.edge_index.shape[1] * epsilon / 2
    feature = data.x.clone() / data.x.sum(1, keepdims=True)
    # remove nan
    for i in range(node_num):
        if torch.sum(feature[i]) != torch.sum(feature[i]):
            feature[i] = 0
    label = data.y
    return Adj.to(device), feature.to(device), label.to(device), edge_epsilon


def load_data(dataset, train_num, device, sample_method, epsilon, test_num, base_num, path_num, house_num, random_num, feature_dim):
    """
    :param dataset: Cora, CiteSeer or PolBlog
    :param train_num: number of training sets
    :param device: device
    """
    if dataset == "Cora":
        node_num = 2708  # 1433 7
        feature_dim = 1433
        class_num = 7
    if dataset == "CiteSeer":
        node_num = 3327  # 3703 6
        feature_dim = 3703
        class_num = 6
    if dataset == "PubMed":
        node_num = 19717  # 500 3
        feature_dim = 500
        class_num = 3
    if dataset == 'PolBlogs':
        node_num = 1490  # 128 2
        feature_dim = feature_dim
        class_num = 2
    if dataset == 'dblp':
        node_num = 17716  # 1639 4
        feature_dim = 1639
        class_num = 4
    if dataset == 'BA':
        node_num = base_num + house_num * 5
        feature_dim = 256
        class_num = 2
    if dataset in ['PolBlogs', 'BA']:
        if dataset == 'PolBlogs':
            Dataset = politic_blog(feature_dim)
        else:
            Dataset = BA(base_num, path_num, house_num, random_num)
    else:
        path = osp.join('datasets/', dataset)
        Dataset = get_dataset(path, dataset)[0].to(device)
    # adjacency matrix, node features and node labels
    adj, X, Y, edge_epsilon = build_data(Dataset, node_num, device, epsilon)
    if sample_method == 'uniform':
        tensor_train_mask, tensor_test_mask = uniform_sampling(node_num, class_num, Y, train_num, test_num)
    if sample_method == 'random':
        tensor_train_mask, tensor_test_mask = random_sampling(node_num, class_num, train_num, test_num)
    return adj, X, Y, edge_epsilon, tensor_train_mask, tensor_test_mask, node_num, feature_dim, class_num

# test
# adj, X, Y, edge_epsilon, train_mask, test_mask, node_num, feature_dim, output_dim = load_data(dataset, train_num, device, 'uniform', epsilon, test_num, base_num, path_num, house_num, random_num, feature_dim) # split dataset
# print(torch.sum(adj))