# DICE
# Meta
# pgd
# MIN MAX
# Static
# Retrained
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
import scipy.sparse as sp
import copy
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device used for computation


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix


class Static_Attack(nn.Module):
    def __init__(self, ori_adj, victim_model, node_num, epsilon, device,
                 lr2, tau, drop_edge_rate_1, drop_edge_rate_2, lambda_1, lambda_2, lambda_3, lambda_4,
                 sample_num, theta):
        super(Static_Attack, self).__init__()
        self.victim = victim_model
        self.mask = nn.Parameter(torch.FloatTensor(int(node_num * (node_num - 1) / 2)))
        # a vector: keep symmetry, save storage space and easy to compute
        self.epsilon = epsilon
        self.node_num = node_num
        self.device = device
        # contrastive and classification
        self.lr2 = lr2
        self.tau = tau
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.sub_adj_1 = self.drop_adj(ori_adj, self.drop_edge_rate_1)
        self.sub_adj_2 = self.drop_adj(ori_adj, self.drop_edge_rate_2)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

        # results
        self.poi_adj = None
        self.theta = theta
        self.best_acc = 1
        # random sampling
        self.sample_num = sample_num

    '''
    def forward(self, adjacency, feature):
        all_adj = torch.ones((node_num, node_num)).to(device)
        adj_bar = all_adj - torch.diag_embed(torch.diag(all_adj)) - adjacency
        adj_bar = adj_bar.to(device)
        attack_adj = adj + (adj_bar - adj).mul(self.Mask)  # 按照扰动率决定分离的边
        return self.Model.con_embed(attack_adj, feature), self.Model(attack_adj, feature)
    '''

    def grad_clipping(self, adj_grad):
        # norm = torch.tensor([0.0], device=self.device)
        norm = (adj_grad ** 2).sum()
        # norm = (abs(adj_grad)).sum()
        norm = norm.sqrt().item()
        # print('norm:', norm)
        if norm > self.theta:
            adj_grad *= (self.theta / norm)
        return adj_grad

    # compute the gradient and renew the mask during training procedure
    def attack(self, ori_adj, ori_features, labels, epochs, train_mask, test_mask, criterion):
        # pro_features = F.normalize(ori_features, p=2, dim=1)
        victim_model = self.victim
        victim_model.eval()  # freeze the parameters except the attack mask
        con_loss_list = []
        cls_loss_list = []
        mis_list = []
        # for t in tqdm(range(epochs), desc="total training"):
        for t in range(epochs):
            self.tau = self.tau * np.sqrt(t + 1) / np.sqrt(t + 2)
            # compute the embedding of poisoned graph
            poi_adj = self.get_poi_adj(ori_adj)
            # normalization
            # adj_norm = self.normalization(poi_adj)
            # output = victim_model(poi_adj, ori_features)
            # compute the loss and gradient of mask
            # cls_loss, con_loss = self.total_loss(poi_adj, ori_adj, ori_features, train_mask, labels, criterion)
            # print("cls_loss:", cls_loss, "con_loss:", con_loss)
            loss = self.total_loss(poi_adj, ori_adj, ori_features, train_mask, labels, criterion)
            # print("cls_loss:", cls_loss, 'con_loss', con_loss)
            # loss = cls_loss + con_loss
            # print('ok')
            # print("epochs:", t, "loss:", loss.item())
            # loss.backward()
            # adj_grad = loss.grad
            #adj_grad_1 = torch.autograd.grad(cls_loss, self.mask)[0]
            #cls_loss, con_loss = self.total_loss(poi_adj, ori_adj, ori_features, train_mask, labels, criterion)
            #adj_grad_2 = torch.autograd.grad(con_loss, self.mask)[0]
            #adj_grad = adj_grad_1 + adj_grad_2
            adj_grad = torch.autograd.grad(loss, self.mask)[0]
            adj_grad = self.grad_clipping(adj_grad)
            # print(adj_grad.data)
            # print("ok")
            # upgrade the mask and use projection operator
            # print(adj_grad.min())
            # lr = - self.lr2
            lr = - self.lr2 / np.sqrt(t + 1)
            # print(self.mask)
            self.mask.data.add_(lr * adj_grad)
            # projection gradient
            self.projection()
            #con_loss_list.append(con_loss)
            #cls_loss_list.append(cls_loss)
        self.random_sample(ori_adj, ori_features, labels, test_mask)
        self.poi_adj = self.get_poi_adj(ori_adj).detach()
        self.best_acc = self.evaluation(ori_adj, train_mask, test_mask, ori_features, labels)
            #mis_list.append(1-self.best_acc)
        # fig = plt.figure(figsize=(6, 4), dpi=100)
        # x = list(range(epochs))
        # plt.plot(x, cls_loss_list)
        # plt.plot(x, con_loss_list)
        # plt.plot(mis_list)
        # plt.legend(['cls_loss', 'con_loss', 'misclassification accuracy'])
        # plt.savefig("loss.pdf")
        # plt.show()
        return self.best_acc

    def get_poi_adj(self, ori_adj):
        complementary = (torch.ones_like(ori_adj) - torch.eye(self.node_num).to(self.device) - ori_adj) - ori_adj
        complementary = complementary.to(self.device)
        # convert mask vector into a mask matrix
        m = torch.zeros((self.node_num, self.node_num)).to(self.device)
        tril_indices = torch.tril_indices(row=self.node_num, col=self.node_num, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.mask
        mask = m + m.t()
        attack_adj = ori_adj + complementary.mul(mask)
        return attack_adj

    def normalization(self, adj):
        adj1 = adj + torch.eye(self.node_num).to(device)
        degree = torch.sum(adj1, axis=1).to(device)
        d_hat = torch.diag(torch.pow(degree, -0.5)).to(device)
        return torch.mm(d_hat, torch.mm(adj1, d_hat))

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def con_exp(self, x):
        return torch.exp(x / self.tau)

    def con_loss(self, z1, z2):
        self_sim = self.con_exp(self.sim(z1, z2))
        between_sim = self.con_exp(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (self_sim.sum(1) + between_sim.sum(1) - self_sim.diag()))

    def final_con_loss(self, z1, z2):
        embed1 = z1
        embed2 = z2
        l1 = self.con_loss(embed1, embed2)
        l2 = self.con_loss(embed2, embed1)
        loss = (l1 + l2) * 0.5
        return loss.mean()

    def con_mid(self, adjacency, feature):
        return self.victim.con_embed(adjacency, feature)

    def drop_adj(self, adjacency, drop_edge_rate):
        rate = 1 - drop_edge_rate
        mid_adj = torch.triu(adjacency)
        mid_adj = mid_adj * rate
        mid_adj = torch.bernoulli(mid_adj)
        return mid_adj + mid_adj.t()

    def total_loss(self, poi_adj, ori_adj, features, train_mask, labels, criterion):
        # contrastive loss
        embed1 = self.victim(poi_adj, features)
        ori_embed = self.victim(ori_adj, features)
        mid_embed1 = self.victim.con_embed(poi_adj, features)
        sub_embed1 = self.con_mid(self.sub_adj_1, features)
        sub_embed2 = self.con_mid(self.sub_adj_2, features)
        con_loss_1 = self.final_con_loss(sub_embed1, sub_embed2)
        con_loss_2 = self.final_con_loss(mid_embed1, sub_embed1)  # / 10
        # classification loss
        train_embed = embed1[train_mask]
        ori_y = labels[train_mask]
        pre_y = ori_embed.max(1)[1]
        train_y = pre_y[train_mask]
        # cls_loss = F.nll_loss(train_embed, train_y)
        cls_loss_1 = criterion(train_embed, ori_y)
        cls_loss_2 = criterion(train_embed, train_y)
        # return self.alpha_1 * con_loss_1 - self.alpha_2 * con_loss_2
        return - (self.lambda_1 * cls_loss_1 + self.lambda_2 * cls_loss_2 + 2e-1 * self.lambda_3 * con_loss_1 + 2e-1 * self.lambda_4 * con_loss_2)
        # return -(2e-1 * self.lambda_3 * con_loss_1 + 2e-1 * self.lambda_4 * con_loss_2)
        # return - (self.lambda_3 * con_loss_1 + self.lambda_4 * con_loss_2)
        # return - (self.lambda_1 * cls_loss_1 + self.lambda_2 * cls_loss_2)

    def projection(self):
        # 控制区间
        projected = torch.clamp(self.mask, 0, 1)
        if projected.sum() > self.epsilon:
            # choose the boundary
            b_min = (self.mask - 1).min()
            b_max = self.mask.max()
            # print(b_min)
            # print(b_max)
            miu = self.bisection(b_min, b_max, err_bound=1e-4)
            new_mask = torch.clamp(self.mask - miu, 0, 1)
        else:
            new_mask = torch.clamp(self.mask, 0, 1)
        self.mask.data.copy_(new_mask)

    # define a function to employ bisection
    def bisection_fun(self, x):
        return torch.clamp(self.mask - x, 0, 1).sum() - self.epsilon

    # bisection method to choose miu
    def bisection(self, b_min, b_max, err_bound):
        miu = b_min
        while (b_max - b_min) > err_bound:
            miu = (b_min + b_max) / 2
            if torch.abs(self.bisection_fun(miu)) < err_bound:
                break
            elif self.bisection_fun(miu) * self.bisection_fun(b_min) < 0:
                b_max = miu
            else:
                b_min = miu
        return miu

    def random_sample(self, ori_adj, ori_features, labels, test_mask):
        best_loss = 10000
        best_acc = 2
        victim_model = self.victim
        victim_model.eval()
        best_s = self.mask
        with torch.no_grad():
            s = self.mask.cpu().detach().numpy()
            # for i in tqdm(range(self.sample_num), desc='sampling...'):
            for i in range(self.sample_num):
                # print(s)
                sample = np.random.binomial(1, s)
                # check the numbers of modified edges
                if sample.sum() > self.epsilon:
                    continue
                self.mask.data.copy_(torch.tensor(sample))
                poi_adj = self.get_poi_adj(ori_adj)
                # adj_norm = self.normalization(poi_adj)
                output = victim_model(poi_adj, ori_features)
                test_embed = output[test_mask]
                predict_y = test_embed.max(1)[1]  # 占比重最大的那个作为类
                acc = torch.eq(predict_y, labels[test_mask]).float().mean()
                # loss = self.total_loss(adj_norm, ori_adj, ori_features, train_mask, labels)
                if best_acc > acc:  # best_loss > loss
                    best_acc = acc
                    # best_loss = loss
                    best_s = sample
            self.mask.data.copy_(torch.tensor(best_s))

    def evaluation(self, ori_adj, train_mask, test_mask, ori_features, labels):
        victim = self.victim
        victim.eval()
        # label_mask = train_mask + test_mask
        label_mask = test_mask
        with torch.no_grad():
            ori_embed = victim(ori_adj, ori_features)
            ori_label_embed = ori_embed[label_mask]
            ori_predict_y = ori_label_embed.max(1)[1]
            ori_accuracy = torch.eq(ori_predict_y, labels[label_mask]).float().mean()
            print("original accuracy:", ori_accuracy)
            adj_norm = self.normalization(self.poi_adj)
            # ori_features = F.normalize(ori_features, p=2, dim=1)
            embed = victim(adj_norm, ori_features)
            # embed = victim(self.poi_adj, ori_features)
            test_embed = embed[label_mask]
            predict_y = test_embed.max(1)[1]  # 占比重最大的那个作为类
            accuracy = torch.eq(predict_y, labels[label_mask]).float().mean()
            # accuracy = torch.eq(predict_y, ori_predict_y).float().mean()
            print("poisoned accuracy:", accuracy)
        return accuracy


class Retrained_Attack(nn.Module):
    def __init__(self, ori_adj, victim_model, node_num, epsilon, device,
                 lr2, tau, drop_edge_rate_1, drop_edge_rate_2, lambda_1, lambda_2, lambda_3, lambda_4,
                 sample_num, theta, attack_lr):
        super(Retrained_Attack, self).__init__()
        self.victim = copy.deepcopy(victim_model)
        self.ori = copy.deepcopy(victim_model)
        self.mask = nn.Parameter(torch.FloatTensor(int(node_num * (node_num - 1) / 2)))
        # a vector: keep symmetry, save storage space and easy to compute
        self.epsilon = epsilon
        self.node_num = node_num
        self.device = device
        # contrastive and classification
        self.lr2 = lr2
        self.tau = tau
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.sub_adj_1 = self.drop_adj(ori_adj, self.drop_edge_rate_1)
        self.sub_adj_2 = self.drop_adj(ori_adj, self.drop_edge_rate_2)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.attack_lr = attack_lr
        # results
        self.poi_adj = None
        self.theta = theta
        self.best_acc = 1
        # random sampling
        self.sample_num = sample_num

    '''
    def forward(self, adjacency, feature):
        all_adj = torch.ones((node_num, node_num)).to(device)
        adj_bar = all_adj - torch.diag_embed(torch.diag(all_adj)) - adjacency
        adj_bar = adj_bar.to(device)
        attack_adj = adj + (adj_bar - adj).mul(self.Mask)  # 按照扰动率决定分离的边
        return self.Model.con_embed(attack_adj, feature), self.Model(attack_adj, feature)
    '''

    def grad_clipping(self, adj_grad):
        norm = torch.tensor([0.0], device=self.device)
        norm = (adj_grad ** 2).sum()
        norm = norm.sqrt().item()
        if norm > self.theta:
            adj_grad *= (self.theta / norm)
        return adj_grad

    # compute the gradient and renew the mask during training procedure
    def attack(self, ori_adj, ori_features, labels, epochs, train_mask, test_mask, criterion):
        victim_model = copy.deepcopy(self.victim)
        optimizer = optim.Adam(victim_model.parameters(), lr=self.attack_lr)
        # victim_model.eval()  # freeze the parameters except the attack mask
        # for t in tqdm(range(epochs), desc="total training"):
        for t in range(epochs):
            self.tau = self.tau * np.sqrt(t + 1) / np.sqrt(t + 2)
            # train W
            victim_model.train()
            # compute the embedding of poisoned graph
            poi_adj = self.get_poi_adj(ori_adj)
            # normalization
            # adj_norm = self.normalization(poi_adj)
            # output = victim_model(adj_norm, ori_features)
            # compute the loss and gradient of mask
            loss = -1 * self.total_loss(poi_adj, ori_adj, ori_features, train_mask, labels, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # optimize M
            victim_model.eval()
            poi_adj = self.get_poi_adj(ori_adj)
            # normalization
            # adj_norm = self.normalization(poi_adj)
            output = victim_model(poi_adj, ori_features)
            # compute the loss and gradient of mask
            loss2 = self.total_loss(poi_adj, ori_adj, ori_features, train_mask, labels, criterion)
            # print('ok')
            # print("epochs:", t, "loss:", loss.item())
            adj_grad = torch.autograd.grad(loss2, self.mask)[0]
            adj_grad = self.grad_clipping(adj_grad)
            # print(adj_grad.data)
            # print("ok")
            # upgrade the mask and use projection operator
            # print(adj_grad.min())
            lr = - self.lr2 * 1 / np.sqrt(t + 1)
            self.mask.data.add_(lr * adj_grad)
            # projection gradient
            self.projection()
        self.random_sample(ori_adj, ori_features, labels, test_mask)
        self.poi_adj = self.get_poi_adj(ori_adj).detach()
        self.best_acc = self.evaluation(ori_adj, train_mask, test_mask, ori_features, labels)
        self.victim = copy.deepcopy(self.ori)
        return self.best_acc

    def get_poi_adj(self, ori_adj):
        complementary = (torch.ones_like(ori_adj) - torch.eye(self.node_num).to(self.device) - ori_adj) - ori_adj
        complementary = complementary.to(self.device)
        # convert mask vector into a mask matrix
        m = torch.zeros((self.node_num, self.node_num)).to(self.device)
        tril_indices = torch.tril_indices(row=self.node_num, col=self.node_num, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.mask
        mask = m + m.t()
        attack_adj = ori_adj + complementary.mul(mask)
        return attack_adj

    def normalization(self, adj):
        adj1 = adj + torch.eye(self.node_num).to(device)
        degree = torch.sum(adj1, axis=1).to(device)
        d_hat = torch.diag(torch.pow(degree, -0.5)).to(device)
        return torch.mm(d_hat, torch.mm(adj1, d_hat))

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def con_exp(self, x):
        return torch.exp(x / self.tau)

    def con_loss(self, z1, z2):
        self_sim = self.con_exp(self.sim(z1, z2))
        between_sim = self.con_exp(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (self_sim.sum(1) + between_sim.sum(1) - self_sim.diag()))

    def final_con_loss(self, z1, z2):
        embed1 = z1
        embed2 = z2
        l1 = self.con_loss(embed1, embed2)
        l2 = self.con_loss(embed2, embed1)
        loss = (l1 + l2) * 0.5
        return loss.mean()

    def con_mid(self, adjacency, feature):
        return self.victim.con_embed(adjacency, feature)

    def drop_adj(self, adjacency, drop_edge_rate):
        rate = 1 - drop_edge_rate
        mid_adj = torch.triu(adjacency)
        mid_adj = mid_adj * rate
        mid_adj = torch.bernoulli(mid_adj)
        return mid_adj + mid_adj.t()

    def total_loss(self, poi_adj, ori_adj, features, train_mask, labels, criterion):
        # contrastive loss
        embed1 = self.victim(poi_adj, features)
        ori_embed = self.victim(ori_adj, features)
        mid_embed1 = self.victim.con_embed(poi_adj, features)
        sub_embed1 = self.con_mid(self.sub_adj_1, features)
        sub_embed2 = self.con_mid(self.sub_adj_2, features)
        con_loss_1 = self.final_con_loss(sub_embed1, sub_embed2)
        con_loss_2 = self.final_con_loss(mid_embed1, sub_embed1)  # / 10
        # classification loss
        train_embed = embed1[train_mask]
        ori_y = labels[train_mask]
        pre_y = ori_embed.max(1)[1]
        train_y = pre_y[train_mask]
        # cls_loss = F.nll_loss(train_embed, train_y)
        cls_loss_1 = criterion(train_embed, ori_y)
        cls_loss_2 = criterion(train_embed, train_y)
        # return self.alpha_1 * con_loss_1 - self.alpha_2 * con_loss_2
        return - (self.lambda_1 * cls_loss_1 + self.lambda_2 * cls_loss_2 + 2e-1 * self.lambda_3 * con_loss_1 +  2e-1 * self.lambda_4 * con_loss_2)

    def projection(self):
        # 控制区间
        projected = torch.clamp(self.mask, 0, 1)
        if projected.sum() > self.epsilon:
            # choose the boundary
            b_min = (self.mask - 1).min()
            b_max = self.mask.max()
            # print(b_min)
            # print(b_max)
            miu = self.bisection(b_min, b_max, err_bound=1e-4)
            new_mask = torch.clamp(self.mask - miu, 0, 1)
        else:
            new_mask = torch.clamp(self.mask, 0, 1)
        self.mask.data.copy_(new_mask)

    # define a function to employ bisection
    def bisection_fun(self, x):
        return torch.clamp(self.mask - x, 0, 1).sum() - self.epsilon

    # bisection method to choose miu
    def bisection(self, b_min, b_max, err_bound):
        miu = b_min
        while (b_max - b_min) > err_bound:
            miu = (b_min + b_max) / 2
            if torch.abs(self.bisection_fun(miu)) < err_bound:
                break
            elif self.bisection_fun(miu) * self.bisection_fun(b_min) < 0:
                b_max = miu
            else:
                b_min = miu
        return miu

    def random_sample(self, ori_adj, ori_features, labels, test_mask):
        best_loss = 10000
        best_acc = 2
        # victim_model = self.victim
        victim_model = copy.deepcopy(self.ori)
        victim_model.eval()
        best_s = self.mask
        with torch.no_grad():
            s = self.mask.cpu().detach().numpy()
            # for i in tqdm(range(self.sample_num), desc='sampling...'):
            for i in range(self.sample_num):
                sample = np.random.binomial(1, s)
                # check the numbers of modified edges
                if sample.sum() > self.epsilon:
                    continue
                self.mask.data.copy_(torch.tensor(sample))
                poi_adj = self.get_poi_adj(ori_adj)
                # adj_norm = self.normalization(poi_adj)
                output = victim_model(poi_adj, ori_features)
                test_embed = output[test_mask]
                predict_y = test_embed.max(1)[1]  # 占比重最大的那个作为类
                acc = torch.eq(predict_y, labels[test_mask]).float().mean()
                # loss = self.total_loss(adj_norm, ori_adj, ori_features, train_mask, labels)
                if best_acc > acc:  # best_loss > loss
                    best_acc = acc
                    # best_loss = loss
                    best_s = sample
            self.mask.data.copy_(torch.tensor(best_s))

    def evaluation(self, ori_adj, train_mask, test_mask, ori_features, labels):
        # victim = self.victim
        victim = copy.deepcopy(self.ori)
        victim.eval()
        # label_mask = train_mask + test_mask
        label_mask = test_mask
        with torch.no_grad():
            ori_embed = victim(ori_adj, ori_features)
            ori_label_embed = ori_embed[label_mask]
            ori_predict_y = ori_label_embed.max(1)[1]
            ori_accuracy = torch.eq(ori_predict_y, labels[label_mask]).float().mean()
            print("original accuracy:", ori_accuracy)
            # adj_norm = self.normalization(self.poi_adj)
            embed = victim(self.poi_adj, ori_features)
            test_embed = embed[label_mask]
            predict_y = test_embed.max(1)[1]  # 占比重最大的那个作为类
            accuracy = torch.eq(predict_y, labels[label_mask]).float().mean()
            # accuracy = torch.eq(predict_y, ori_predict_y).float().mean()
            print("poisoned accuracy:", accuracy)
        return accuracy


class PGD_Attack(nn.Module):
    def __init__(self, ori_adj, victim_model, node_num, epsilon, device, lr2, sample_num, theta):
        super(PGD_Attack, self).__init__()
        self.victim = victim_model
        self.mask = nn.Parameter(torch.FloatTensor(int(node_num * (node_num - 1) / 2)))
        # a vector: keep symmetry, save storage space and easy to compute
        self.epsilon = epsilon
        self.node_num = node_num
        self.device = device
        # contrastive and classification
        self.lr2 = lr2

        # results
        self.poi_adj = None
        self.theta = theta
        self.best_acc = 1
        # random sampling
        self.sample_num = sample_num

    '''
    def forward(self, adjacency, feature):
        all_adj = torch.ones((node_num, node_num)).to(device)
        adj_bar = all_adj - torch.diag_embed(torch.diag(all_adj)) - adjacency
        adj_bar = adj_bar.to(device)
        attack_adj = adj + (adj_bar - adj).mul(self.Mask)  # 按照扰动率决定分离的边
        return self.Model.con_embed(attack_adj, feature), self.Model(attack_adj, feature)
    '''

    def grad_clipping(self, adj_grad):
        norm = torch.tensor([0.0], device=self.device)
        norm = (adj_grad ** 2).sum()
        norm = norm.sqrt().item()
        if norm > self.theta:
            adj_grad *= (self.theta / norm)
        return adj_grad

    # compute the gradient and renew the mask during training procedure
    def attack(self, ori_adj, ori_features, labels, epochs, train_mask, test_mask, criterion):
        victim_model = self.victim
        victim_model.eval()  # freeze the parameters except the attack mask
        # for t in tqdm(range(epochs), desc="total training"):
        for t in range(epochs):
            # compute the embedding of poisoned graph
            poi_adj = self.get_poi_adj(ori_adj)
            # normalization
            # adj_norm = self.normalization(poi_adj)
            # output = victim_model(poi_adj, ori_features)
            # compute the loss and gradient of mask
            loss = self.total_loss(poi_adj, ori_adj, ori_features, train_mask, labels, criterion)
            # print('ok')
            # print("epochs:", t, "loss:", loss.item())
            adj_grad = torch.autograd.grad(loss, self.mask)[0]
            adj_grad = self.grad_clipping(adj_grad)
            # print(adj_grad.data)
            # print("ok")
            # upgrade the mask and use projection operator
            # print(adj_grad.min())
            lr = - self.lr2 * 1 / np.sqrt(t + 1)
            # print(self.mask)
            self.mask.data.add_(lr * adj_grad)
            # projection gradient
            self.projection()
        self.random_sample(ori_adj, ori_features, labels, test_mask)
        self.poi_adj = self.get_poi_adj(ori_adj).detach()
        self.best_acc = self.evaluation(ori_adj, train_mask, test_mask, ori_features, labels)
        return self.best_acc

    def get_poi_adj(self, ori_adj):
        complementary = (torch.ones_like(ori_adj) - torch.eye(self.node_num).to(self.device) - ori_adj) - ori_adj
        complementary = complementary.to(self.device)
        # convert mask vector into a mask matrix
        m = torch.zeros((self.node_num, self.node_num)).to(self.device)
        tril_indices = torch.tril_indices(row=self.node_num, col=self.node_num, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.mask
        mask = m + m.t()
        attack_adj = ori_adj + complementary.mul(mask)
        return attack_adj

    def normalization(self, adj):
        adj1 = adj + torch.eye(self.node_num).to(device)
        degree = torch.sum(adj1, axis=1).to(device)
        d_hat = torch.diag(torch.pow(degree, -0.5)).to(device)
        return torch.mm(d_hat, torch.mm(adj1, d_hat))

    def con_mid(self, adjacency, feature):
        return self.victim.con_embed(adjacency, feature)

    def drop_adj(self, adjacency, drop_edge_rate):
        rate = 1 - drop_edge_rate
        mid_adj = torch.triu(adjacency)
        mid_adj = mid_adj * rate
        mid_adj = torch.bernoulli(mid_adj)
        return mid_adj + mid_adj.t()

    def total_loss(self, poi_adj, ori_adj, features, train_mask, labels, criterion):
        # contrastive loss
        embed1 = self.victim(poi_adj, features)
        ori_embed = self.victim(ori_adj, features)

        # classification loss
        train_embed = embed1[train_mask]
        pre_y = ori_embed.max(1)[1]
        train_y = pre_y[train_mask]
        cls_loss_2 = criterion(train_embed, train_y)
        return - cls_loss_2

    def projection(self):
        # 控制区间
        projected = torch.clamp(self.mask, 0, 1)
        if projected.sum() > self.epsilon:
            # choose the boundary
            b_min = (self.mask - 1).min()
            b_max = self.mask.max()
            # print(b_min)
            # print(b_max)
            miu = self.bisection(b_min, b_max, err_bound=1e-4)
            new_mask = torch.clamp(self.mask - miu, 0, 1)
        else:
            new_mask = torch.clamp(self.mask, 0, 1)
        self.mask.data.copy_(new_mask)

    # define a function to employ bisection
    def bisection_fun(self, x):
        return torch.clamp(self.mask - x, 0, 1).sum() - self.epsilon

    # bisection method to choose miu
    def bisection(self, b_min, b_max, err_bound):
        miu = b_min
        while (b_max - b_min) > err_bound:
            miu = (b_min + b_max) / 2
            if torch.abs(self.bisection_fun(miu)) < err_bound:
                break
            elif self.bisection_fun(miu) * self.bisection_fun(b_min) < 0:
                b_max = miu
            else:
                b_min = miu
        return miu

    def random_sample(self, ori_adj, ori_features, labels, test_mask):
        best_loss = 10000
        best_acc = 2
        victim_model = self.victim
        victim_model.eval()
        best_s = self.mask
        with torch.no_grad():
            s = self.mask.cpu().detach().numpy()
            # for i in tqdm(range(self.sample_num), desc='sampling...'):
            for i in range(self.sample_num):
                # print(s)
                sample = np.random.binomial(1, s)
                # check the numbers of modified edges
                if sample.sum() > self.epsilon:
                    continue
                self.mask.data.copy_(torch.tensor(sample))
                poi_adj = self.get_poi_adj(ori_adj)
                # adj_norm = self.normalization(poi_adj)
                output = victim_model(poi_adj, ori_features)
                test_embed = output[test_mask]
                predict_y = test_embed.max(1)[1]  # 占比重最大的那个作为类
                acc = torch.eq(predict_y, labels[test_mask]).float().mean()
                # loss = self.total_loss(adj_norm, ori_adj, ori_features, train_mask, labels)
                if best_acc > acc:  # best_loss > loss
                    best_acc = acc
                    # best_loss = loss
                    best_s = sample
            self.mask.data.copy_(torch.tensor(best_s))

    def evaluation(self, ori_adj, train_mask, test_mask, ori_features, labels):
        victim = self.victim
        victim.eval()
        # label_mask = train_mask + test_mask
        label_mask = test_mask
        with torch.no_grad():
            ori_embed = victim(ori_adj, ori_features)
            ori_label_embed = ori_embed[label_mask]
            ori_predict_y = ori_label_embed.max(1)[1]
            ori_accuracy = torch.eq(ori_predict_y, labels[label_mask]).float().mean()
            print("original accuracy:", ori_accuracy)
            adj_norm = self.normalization(self.poi_adj)
            embed = victim(adj_norm, ori_features)
            # embed = victim(self.poi_adj, ori_features)
            test_embed = embed[label_mask]
            predict_y = test_embed.max(1)[1]  # 占比重最大的那个作为类
            accuracy = torch.eq(predict_y, labels[label_mask]).float().mean()
            # accuracy = torch.eq(predict_y, ori_predict_y).float().mean()
            print("poisoned accuracy:", accuracy)
        return accuracy

