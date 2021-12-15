from typing_extensions import Concatenate
import dgl
import torch
from torch._C import _llvm_enabled
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from dgl.nn import SAGEConv

label
item user
Concat
MLP

pred
\sigma (pred - label)^2 + \sigma (pred_i - pred_j)^2 + \lambda 
loss function

positive
negative



class GraphSAGE(nn.module):
    # here h_feats_0 is number_user_features, h_feats_1 is number_item_features
    def __init__(self, in_feats, h_feats_0, h_feats_1, rel_names):
        super(GraphSAGE, self).__init__()
        # self.conv1 = GraphSAGE(h_feats_0, h_feats_0, 'mean')
        # self.conv2 = GraphSAGE(h_feats_0, h_feats_0, 'mean')
        # self.conv3 = GraphSAGE(h_feats_1, h_feats_1, 'mean')
        # self.conv4 = GraphSAGE(h_feats_1, h_feats_1, 'mean')

        self.conv0 = nn.GraphConv(h_feats_0, h_feats_0)
        self.conv1 = nn.GraphConv(h_feats_1, h_feats_1)
        self.CONV0 = nn.HeteroGraphConv({'evaluate': self.conv0, 'evaluated': self.conv1}, aggregate = 'sum')
        self.CONV1 = nn.HeteroGraphConv({'evaluate': self.conv0, 'evaluated': self.conv1}, aggregate = 'sum')

        '''
        self.conv1 = nn.HeteroGraphConv({
            rel: nn.GraphConv(in_feats, h_feats_0)
            for rel in rel_names}, aggregate = 'sum'
        )

        self.CONV0 = ({self.conv1, self.conv3})
        self.CONV1 = ({self.conv2, self.conv4})
        '''


    def forward(self, g, inputs):
        with g.local_scope():
            h = self.CONV0(g, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.CONV1(g, h)
            return h


class MLPPredictor(nn.module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats, 1)
    
    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W1(h).squeeze(1)}
    
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.data['score']

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels) + F.mse_loss(scores, scores)

from sklearn.metrics import roc_auc_score
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    print(scores)
    print(scores.shape)
    print('acc = ', labels == scores / (scores.shape[0]))
    print('roc_auc_score = ', roc_auc_score(labels, scores))
    return roc_auc_score(labels, scores)

train_g = []
number_user_features = 79
number_item_features = 152 - 79
model = GraphSAGE(train_g.ndata['features'].shape[1], number_user_features, number_item_features)
h = model(train_g, number_user_features + number_item_features, number_user_features, number_item_features)

pred = MLPPredictor(number_user_features + number_item_features)

