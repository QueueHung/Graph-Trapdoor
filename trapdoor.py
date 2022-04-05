from networkx.algorithms.assortativity import neighbor_degree
from networkx.classes.function import neighbors
from GraphModule.gcn import MyGCN, MyGCN2
from GraphModule.nettack import Nettack

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from numpy.random import default_rng
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.utils import accuracy, preprocess
from deeprobust.graph.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(7414)
random.seed(7414)
rng = default_rng(7414)

def FindOneHopNeighbor(target_node, adj):
    one_hop_neighbors = set()
    for i in np.vstack(np.nonzero(adj)).T:
        if i[0] == target_node:
            one_hop_neighbors.add(i[1])
    return np.array(list(one_hop_neighbors)) 
    
def FindTwoHopNeighbor(target_node, adj):
    one_hop_neighbors = set()
    two_hop_neighbors = set()
    for i in np.vstack(np.nonzero(adj)).T:
        if i[0] == target_node:
            one_hop_neighbors.add(i[1])
    for j in one_hop_neighbors:
        for i in np.vstack(np.nonzero(adj)).T:
            if i[0] == j:
                two_hop_neighbors.add(i[1])
    two_hop_neighbors.remove(target_node)
    return np.union1d( list(two_hop_neighbors), list(one_hop_neighbors) )
    
def ChooseGCN(GCN_index = 1):
    '''
    0: original GCN
    '''
    if GCN_index == 0:
        target_gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()+1), device=device)
        target_gcn.fit(features, adj, labels, idx_train, idx_value)
    elif GCN_index == 1:
        target_gcn = MyGCN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()+1), device=device)
        target_gcn.fit(features, adj, labels, idx_train, idx_trap, idx_value)
    elif GCN_index == 2:
        target_gcn = MyGCN2(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()+1), device=device)
        target_gcn.fit(features, adj, labels, idx_train, idx_value)
    return target_gcn


data = Dataset(root='Dataset', name='cora', seed=15)
# number of node = 2485
# number of features of each node = 1433
adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_value, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_trap = rng.choice(idx_train, size = int(0.01*idx_train.shape[0]))

print(idx_trap)
labels[idx_trap] = labels.max()+1
idx_train = np.setdiff1d(idx_train, idx_trap)

adj, features, labels = preprocess(adj, features, labels)

''' choose one '''
target_gcn = ChooseGCN()
print("test set:", target_gcn.test(idx_test))
''' choose one '''

model = PGDAttack(model=target_gcn, nnodes=adj.shape[0], loss_type='CE', device=device)
model = model.to(device)

model.attack(features, adj, labels, idx_train, 10)
modified_adj = model.modified_adj.to(device)
modified_edge = torch.nonzero( torch.logical_xor(modified_adj.cpu(), adj) ).numpy()

print(modified_edge.flatten())

target_gcn.initialize()
target_gcn.fit(features, modified_adj, labels, idx_train, idx_trap, idx_trap)
print(target_gcn.test(idx_test))
print()
for u in np.unique(modified_edge.flatten()) :
    print("u:", u)
    two_hop = FindTwoHopNeighbor(u, adj.numpy())
    print("Two hop neighbor:", two_hop, sep='\n')
    print("Intersetction between two hop and idx_trap",np.intersect1d(two_hop, idx_trap), sep="\n")
    print()
