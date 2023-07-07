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
from data_processing import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device used for computation


# train num of datasets split in training sets
# epsilon = 0.05
# train_num = 40  # single class
# test_num = 1000
# dataset = 'Cora'  # dataset
# # BA parameter
# base_num = 1000
# path_num = 5
# house_num = 200
# random_num = 10
# test_num = 1000
# # polblogs
# feature_dim = 256
# adj, X, Y, edge_epsilon, train_mask, test_mask, node_num, feature_dim, output_dim = load_data(dataset, train_num,
#                                                                                                device, 'uniform',
#                                                                                                epsilon,test_num, base_num, path_num, house_num, random_num, feature_dim)  # split dataset

# print(adj.shape)
# training GCN
# dimension of hidden embedding in GNN
# gcn_hidden = 32
# dimension of GCN output and MLP input
# con_num = 128
# dimension of hidden embedding in MLP
# mlp_hidden = 32
# training procedure
# epochs1 = 200
# learning_rate1 = 0.01
# weight_decay = 1e-4


def normalization(adjacency):
    """
    :param adjacency:
    :return: normalized adjacency
    """
    adjacency1 = adjacency + torch.eye(adjacency.shape[0]).to(device)
    degree = torch.sum(adjacency1, axis=1).to(device)
    d_hat = torch.diag(torch.pow(degree, -0.5)).to(device)
    return torch.mm(d_hat, torch.mm(adjacency1, d_hat))


class GraphConvolution(nn.Module):
    """
    GCN layer
    """

    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        initial weight
        """
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_features):
        """
        :param adjacency: adj matrix
        :param input_features: initial features or the embedding at previous layer
        :return: node embedding
        """
        # input_features = F.normalize(input_features, p=2, dim=1)
        support = torch.mm(input_features, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class GCN(nn.Module):
    """
    graph convolution network
    """

    def __init__(self, input_dim, output_dim, gcn_hidden, con_num, mlp_hidden):
        super(GCN, self).__init__()
        num_classes = output_dim
        self.gcn1 = GraphConvolution(input_dim, gcn_hidden)
        self.gcn2 = GraphConvolution(gcn_hidden, con_num)
        self.FC1 = torch.nn.Linear(con_num, mlp_hidden, bias=False)
        self.FC2 = torch.nn.Linear(mlp_hidden, num_classes, bias=False)

    def forward(self, adjacency, feature):
        """
        :param adjacency: matrix
        :param feature:  feature
        :return: final embedding
        """
        # adjacency = normalization(adjacency).to(device)
        embed1 = F.relu(self.con_embed(adjacency, feature))
        # embed2 = self.FC1(embed1)
        embed2 = F.relu(self.FC1(embed1))
        results = self.FC2(embed2)
        return results
        # return embed2
        # return self.con_embed(adjacency, feature)

    def con_embed(self, adjacency, feature):
        adjacency = normalization(adjacency).to(device)
        h = F.relu(self.gcn1(adjacency, feature))
        # h = F.sigmoid(self.gcn1(adjacency, feature))
        # h = F.tanh(self.gcn1(adjacency, feature))
        # h = F.leaky_relu(self.gcn1(adjacency, feature))
        embed1 = self.gcn2(adjacency, h)
        return embed1


# class One_Layer_GCN(nn.Module):
#     """
#     graph convolution network
#     """
#
#     def __init__(self, input_dim, output_dim, gcn_hidden, con_num, mlp_hidden):
#         super(One_Layer_GCN, self).__init__()
#         num_classes = output_dim
#         self.gcn1 = GraphConvolution(input_dim, con_num)
#         self.FC1 = torch.nn.Linear(con_num, mlp_hidden, bias=False)
#         self.FC2 = torch.nn.Linear(mlp_hidden, num_classes, bias=False)
#
#     def forward(self, adjacency, feature):
#         """
#         :param adjacency: matrix
#         :param feature:  feature
#         :return: final embedding
#         """
#         # adjacency = normalization(adjacency).to(device)
#         embed1 = F.relu(self.con_embed(adjacency, feature))
#         # embed2 = self.FC1(embed1)
#         embed2 = F.relu(self.FC1(embed1))
#         results = self.FC2(embed2)
#         return results
#         # return embed2
#         # return self.con_embed(adjacency, feature)
#
#     def con_embed(self, adjacency, feature):
#         adjacency = normalization(adjacency).to(device)
#         embed1 = self.gcn1(adjacency, feature)
#         # embed1 = self.gcn2(adjacency, h)
#         return embed1
#
#
# class Three_Layers_GCN(nn.Module):
#     """
#     graph convolution network
#     """
#
#     def __init__(self, input_dim, output_dim, gcn_hidden, con_num, mlp_hidden):
#         super(Three_Layers_GCN, self).__init__()
#         num_classes = output_dim
#         self.gcn1 = GraphConvolution(input_dim, gcn_hidden)
#         self.gcn2 = GraphConvolution(gcn_hidden, gcn_hidden)
#         self.gcn3 = GraphConvolution(gcn_hidden, con_num)
#         self.FC1 = torch.nn.Linear(con_num, mlp_hidden, bias=False)
#         self.FC2 = torch.nn.Linear(mlp_hidden, num_classes, bias=False)
#
#     def forward(self, adjacency, feature):
#         """
#         :param adjacency: matrix
#         :param feature:  feature
#         :return: final embedding
#         """
#         # adjacency = normalization(adjacency).to(device)
#         embed1 = F.relu(self.con_embed(adjacency, feature))
#         # embed2 = self.FC1(embed1)
#         embed2 = F.relu(self.FC1(embed1))
#         results = self.FC2(embed2)
#         return results
#         # return embed2
#         # return self.con_embed(adjacency, feature)
#
#     def con_embed(self, adjacency, feature):
#         adjacency = normalization(adjacency).to(device)
#         h = F.relu(self.gcn1(adjacency, feature))
#         embed1 = F.relu(self.gcn2(adjacency, h))
#         embed2 = self.gcn3(adjacency, embed1)
#         return embed2


class SAGELayer(nn.Module):
    """
    GCN layer
    """

    def __init__(self, input_dim, output_dim, use_bias=False):
        super(SAGELayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight1 = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight2 = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        initial weight
        """
        init.kaiming_uniform_(self.weight1)
        init.kaiming_uniform_(self.weight2)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_features):
        """
        :param adjacency: adj matrix
        :param input_features: initial features or the embedding at previous layer
        :return: node embedding
        """
        support = torch.mm(input_features, self.weight1)
        output = torch.mm(adjacency, support) + torch.mm(input_features, self.weight2)
        if self.use_bias:
            output += self.bias
        return output


class GraphSAGE(nn.Module):
    """
    graph convolution network
    """

    def __init__(self, input_dim, output_dim, gcn_hidden, con_num, mlp_hidden):
        super(GraphSAGE, self).__init__()
        num_classes = output_dim
        self.gcn1 = SAGELayer(input_dim, gcn_hidden)
        self.gcn2 = SAGELayer(gcn_hidden, con_num)
        self.FC1 = torch.nn.Linear(con_num, mlp_hidden, bias=False)
        self.FC2 = torch.nn.Linear(mlp_hidden, num_classes, bias=False)

    def forward(self, adjacency, feature):
        """
        :param adjacency: matrix
        :param feature:  feature
        :return: final embedding
        """
        # adjacency = normalization(adjacency).to(device)
        embed1 = F.relu(self.con_embed(adjacency, feature))
        # embed2 = self.FC1(embed1)
        embed2 = F.relu(self.FC1(embed1))
        results = self.FC2(embed2)
        return results
        # return embed2

    def con_embed(self, adjacency, feature):
        adjacency = normalization(adjacency).to(device)
        h = F.relu(self.gcn1(adjacency, feature))
        embed1 = self.gcn2(adjacency, h)
        return embed1


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True, training=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        # self.training = training

        # define parameters
        self.weight = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        init.xavier_uniform_(self.weight.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim, 1)))
        init.xavier_uniform_(self.a.data, gain=1.414)
        # define activation function
        self.leaky_ReLu = nn.LeakyReLU(self.alpha)

    def forward(self, adjacency, features):
        h = torch.mm(features, self.weight)
        node_num = h.size()[0]
        a_input = torch.cat([h.repeat(1, node_num).view(node_num * node_num, -1), h.repeat(node_num, 1)], dim=1) \
            .view(node_num, -1, 2 * self.output_dim)
        # [N, N, 2 * output_dim]
        e = self.leaky_ReLu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N]
        # set unconnected edges -inf
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adjacency > 0, e, zero_vec)  # attention score of nodes
        # attention = e.mul(adjacency)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = attention.mul(adjacency)
        h_prime = torch.matmul(attention, h)
        # h_prime = torch.matmul(adjacency, h_prime)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


# introduce multi-head attention to construct GAT
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, dropout, alpha, num_heads):
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = num_class
        self.dropout = dropout
        self.alpha = alpha
        self.num_heads = num_heads
        # construct multi_head attention GNN
        self.attentions = [GraphAttentionLayer(self.input_dim, self.hidden_dim, self.dropout, self.alpha, concat=True)
                           for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden_dim * num_heads, num_class, dropout, alpha, concat=True)
        self.FC1 = nn.Linear(num_class, num_class, bias=False)
        self.FC2 = nn.Linear(num_class, num_class, bias=False)

    def forward(self, adjacency, features):
        # x = F.dropout(features, self.dropout, training=self.training)
        # x = torch.cat([att(adjacency, x) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(adjacency, x))
        x = self.con_embed(adjacency, features)
        x = F.elu(self.FC1(x))
        x = F.elu(self.FC2(x))
        results = F.log_softmax(x, dim=1)
        return results

    def con_embed(self, adjacency, features):
        adjacency = normalization(adjacency).to(device)
        x = F.dropout(features, self.dropout, training=self.training)
        x = torch.cat([att(adjacency, x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(adjacency, x))
        return x


# model can not be viewd as a local variable
def train_GNN(train_mask, test_mask, X, Y, adj, model, epochs, criterion, optimizer):
    """
    train a GNN
    :param train_mask: split data into training set
    :param test_mask: split data into test set
    :param X: node feature
    :param Y: node lebel
    :param adj: adjacency matrix
    :param model: GNN
    :param epochs: number of epochs
    :param criterion: loss
    :param optimizer: Adam or SGD
    :return: trained model, loss history
    """
    loss_history = []
    test_acc_history = []
    model.train()
    train_y = Y[train_mask]
    # for epoch in tqdm(range(epochs), desc='training GNN...'):
    for epoch in range(epochs):
        embed = model(adj, X)
        train_embed = embed[train_mask]
        loss = criterion(train_embed, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = evaluation(model, train_mask, adj, X, Y)
        test_acc = evaluation(model, test_mask, adj, X, Y)
        loss_history.append(loss.item())
        # val_acc_history.append(val_acc.item())
        test_acc_history.append(test_acc.item())
        # print("Epoch{:03d}:Loss{:.4f}, TrainAcc{:.4},TestAcc{:.4f}".format(epoch, loss.item(), train_acc.item(),
        #                                                                    test_acc.item()))
    print("GCN already trained! train accuracy is,", train_acc.item(), 'test accuracy is,', test_acc.item())
    return model, train_acc


def evaluation(model, mask, adj, X, Y):
    """
    calculate the accuracy
    :param model: GNN
    :param mask: test or train mask
    :param adj: adjacency matrix
    :param X: node feature
    :param Y: node label
    :return: accuracy
    """
    model.eval()
    with torch.no_grad():
        embed = model(adj, X)
        test_embed = embed[mask]
        predict_y = test_embed.max(1)[1]  # 占比重最大的那个作为类
        accuracy = torch.eq(predict_y, Y[mask]).float().mean()
    return accuracy

# test
# victim_model = GCN(input_dim=feature_dim, output_dim=output_dim).to(device)  # hyper
# victim_model = GAT(input_dim=feature_dim, hidden_dim=16, num_class=output_dim, dropout=0.1, alpha=0.1, num_heads=1).to(device)
# victim_model = GraphSAGE(input_dim=feature_dim, output_dim=output_dim).to(device)  # hyper
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer1 = optim.Adam(victim_model.parameters(), lr=learning_rate1, weight_decay=weight_decay)
# victim_model, train_acc = train_GNN(train_mask, test_mask, X, Y, adj, victim_model, epochs1, criterion, optimizer1)
