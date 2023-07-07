import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_processing import load_data
from Victim_Model import GCN, train_GNN, GAT, GraphSAGE
from Attack_Method import Static_Attack, Retrained_Attack, PGD_Attack
import copy
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device used for computation
# device = 'cpu'
# train num of datasets split in training sets
train_num = 20  # single class
test_num = 1000
dataset = 'Cora'  # dataset
# polblogs
feature_dim = 512
# parameters during training GCN
# dimension of hidden embedding in GNN
gcn_hidden = 32  # 32
# dimension of GCN output and MLP input
con_num = 64  # 128
# dimension of hidden embedding in MLP
mlp_hidden = 32  # 16
# training procedure
epochs1 = 150  # 150
learning_rate1 = 0.01  # 0.01
weight_decay = 1e-4  # 1e-4
# parameters during attacking GCN
# attack_num_hidden = 32
# clipping threhold
theta_0 = 5e-3
learning_rate2 = 200  # 500
epochs2 = 150  # 150
epsilon = 0.01
# contrastive parameter
tau = 0.4  # 0.8
drop_edge_rate_1 = 0
drop_edge_rate_2 = 0.8  # 0.7
# total loss
lambda_1 = 0.1  # 0.9
lambda_2 = 0.9  # poi
lambda_3 = 0.1  # 0.9
lambda_4 = 0.9  # poi
# random_sample
sample_num = 50
# BA parameter
base_num = 1000
path_num = 5
house_num = 200
random_num = 10
seed = 1219  # cora :1219
attack_lr = 0.01
iter = 15


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


pgd = []
tag = []
for num in tqdm([1219], desc="total training..."):
    setup_seed(num)
    adj, X, Y, edge_epsilon, train_mask, test_mask, node_num, feature_dim, output_dim \
        = load_data(dataset, train_num, device, 'uniform', epsilon, test_num, base_num, path_num, house_num, random_num,
                    feature_dim)  # split dataset
    victim_model = GCN(feature_dim, output_dim, gcn_hidden, con_num, mlp_hidden).to(device)  # hyper
    # victim_model = GAT(input_dim=feature_dim, hidden_dim=8, num_class=output_dim, dropout=0.1, alpha=0.1, num_heads=2).to(device)
    # victim_model = GraphSAGE(feature_dim, output_dim, gcn_hidden, con_num, mlp_hidden).to(device)  # hyper
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer1 = optim.Adam(victim_model.parameters(), lr=learning_rate1, weight_decay=weight_decay)
    victim_model, train_acc = train_GNN(train_mask, test_mask, X, Y, adj, victim_model, epochs1, criterion, optimizer1)

    # attack
    # PGD
    #'''
    for n in tqdm([5], desc='training...'):
        best_test_acc = 1
        theta = theta_0 * n / 5
        for _ in range(iter):
            try:
                attack_model = PGD_Attack(adj, victim_model, node_num, edge_epsilon * n, device, learning_rate2,
                                          sample_num, theta).to(device)
                acc = attack_model.attack(adj, X, Y, epochs2, train_mask, test_mask, criterion)
            except Exception as e:
                print('warning:', e)
            if best_test_acc > acc:
                best_test_acc = acc
        print('-------------------data:', dataset, 'seed:', num, '****EDGE,', n, 'BEST PGD:', best_test_acc, '----------------')
        # pgd.append(1 - best_test_acc.item())
    #'''
    # ours-static
    for n in tqdm([5], desc='training...'):
        theta = theta_0 * n / 5
        best_test_acc = 1
        # for _ in tqdm(range(iter), desc='training...'):
        for _ in range(iter):
            try:
                # drop_edge_rate_2 = d1
                attack_model = Static_Attack(adj, victim_model, node_num, edge_epsilon * n, device,
                                             learning_rate2, tau, drop_edge_rate_1, drop_edge_rate_2, lambda_1,
                                             lambda_2, lambda_3,
                                             lambda_4,
                                             sample_num, theta).to(device)
                acc = attack_model.attack(adj, X, Y, epochs2, train_mask, test_mask, criterion)
            except Exception as e:
                print('warning:', e)
                continue
            if best_test_acc > acc:
                best_test_acc = acc
        print('-------------------data:', dataset, '****EDGE,', n, 'BEST OURS:', best_test_acc, '----------------')
        # tag.append(1 - best_test_acc.item())

    # MinMax
    '''
    for n in [1, 5, 10]:
        best_test_acc = 1
        for _ in range(iter):
            attack_model = MinMax_Attack(adj, victim_model, node_num, edge_epsilon * n, device, learning_rate2, sample_num, theta, attack_lr).to(device)
            acc = attack_model.attack(adj, X, Y, epochs2, train_mask, test_mask, criterion)
            if best_test_acc > acc:
                best_test_acc = acc
        print('-------------------data:', dataset, '****EDGE,', n, 'BEST MinMax:', best_test_acc,'----------------')
        torch.cuda.empty_cache()
    '''

    '''
    # ours-retrained
    for n in [5, 10, 15, 20, 25, 30]:
        theta = theta_0 * n / 5
        best_test_acc = 1
        for _ in range(iter):
            attack_model = Retrained_Attack(adj, victim_model, node_num, edge_epsilon * n, device,
                                         learning_rate2, tau, drop_edge_rate_1, drop_edge_rate_2, lambda_1, lambda_2, lambda_3,
                                         lambda_4,
                                         sample_num, theta, attack_lr).to(device)
            acc = attack_model.attack(adj, X, Y, epochs2, train_mask, test_mask, criterion)
            if best_test_acc > acc:
                best_test_acc = acc
        print('-------------------data:', dataset, '****EDGE,', n, 'BEST OURS-RETRAINED:', best_test_acc,'----------------')
    '''
