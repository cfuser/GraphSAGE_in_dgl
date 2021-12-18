import csv
import torch
import dgl

from dgl.data import dgl_dataset
from torch._C import device

f = open('D:\subject\graduate\graph computation\data\offline\\train0.csv', 'r')
reader = csv.reader(f)

M = 79
N = 152

record = []

node_num = 0
# user = []
user = {}
item = {}

features = []
user_features = []
item_features = []
labels = []
labeled = []

_edge = 0
for i in reader:
    #record.append(i)
    temp = {}
    temp['uuid'] = int(i[0])
    temp['visit_time'] = int(i[1])
    temp['user_id'] = int(i[2])
    temp['item_id'] = int(i[3])
    features = i[4].split()
    features = [float(_) for _ in features]
    temp['features'] = features
    temp['label'] = int(i[5])
    if (temp['label'] != -1):
        labeled.append(_edge)
    _edge = _edge + 1
    record.append(temp)
    if not(int(i[2]) in user):
        user[int(i[2])] = node_num
        node_num = node_num + 1
        user_features.append(features[M:])
    else:
        # detect whether the feature of user is same or not
        pass

# print(record[0])
user_idx_begin = 0
item_idx_begin = node_num

# link[0] is user, link[1] is item
link = [[], []]
for i in record:
    # print(i['item_id'])
    if not(i['item_id'] in item):
        item[i['item_id']] = node_num
        node_num = node_num + 1
        item_features.append(i['features'][:M])
    else:
        # detect whether the featuser of item is same or not
        pass
    # print(user[i['user_id']])
    # print(item[i['item_id']])
    # break
    # link.append([user[i['user_id']], item[i['item_id']]])
    link[0].append(user[i['user_id']])
    link[1].append(item[i['item_id']] - item_idx_begin)
link_tensor = torch.tensor(link)

graph_data = {('user', 'evaluate', 'item'): (link_tensor[0], link_tensor[1]),
                ('item', 'evaluated', 'user'): (link_tensor[1], link_tensor[0])
                }
g = dgl.heterograph(graph_data)
print(g)
g = g.to('cuda:0')
# user_features = user_features.to('cuda:0')
# item_features = item_features.to('cuda:0')

number_user_features = 73
number_item_features = 152 - number_user_features
# print(torch.tensor(user_features))
_min = torch.min(torch.tensor(user_features), dim = 0)
_max = torch.max(torch.tensor(user_features), dim = 0)
# print(_min)
# print(_max)
_dif = _max[0] - _min[0]
_nonzero_idx = torch.nonzero(_dif == 0)

# torch.set_printoptions(precision=8)

# print(_dif)
# print(_nonzero_idx)
_dif[_nonzero_idx] = 1
# print(_dif)
_temp = (torch.tensor(user_features) - _min[0]) / _dif
user_features = _temp.numpy().tolist()
# print((torch.tensor(user_features) - _min[0]) / _dif)
# _max = torch.max(_temp, dim = 0)
# print(_max)
_min = torch.min(torch.tensor(item_features), dim = 0)
_max = torch.max(torch.tensor(item_features), dim = 0)
_dif = _max[0] - _min[0]
_nonzero_idx = torch.nonzero(_dif == 0)
_dif[_nonzero_idx] = 1
_temp = (torch.tensor(item_features) - _min[0]) / _dif
item_features = _temp.numpy().tolist()

for i in range(number_user_features):
    pass

for i in record:
    features.append(i['features'])
    # item_features.append(i['features'][:M])
    # user_features.append(i['features'][M:])
    labels.append(i['label'])

#print(g)
print(torch.tensor(user_features).size())
print(torch.tensor(item_features).size())
g.nodes['user'].data['features'] = torch.tensor(user_features).to('cuda:0')
g.nodes['item'].data['features'] = torch.tensor(item_features).to('cuda:0')

#g.ndata['features'] = torch.tensor(item_features)
#g.edata['features'] = torch.tensor(labels)
g = g.to('cuda:0')
# user_features = user_features.to(device = torch.device('cuda'))
# item_features = item_features.to(device = torch.device('cuda'))
print(g)

import torch.nn as nn
import dgl.function as fn

'''
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h   #一次性为所有节点类型的 'h'赋值
            graph.apply_edges(fn.u_dot_v('features', 'features', 'score'), etype=etype)
            return graph.edges[etype].data['score']
'''

class MLPPredictor(nn.Module):
    def __init__(self, in_features_0, in_features_1, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features_0 + in_features_1, out_classes)
    def apply_edges(self, edges):
        # a = input()
        # print(edges.src)
        # print(edges.dst)
        '''
        if (edges.src['user_features']):
            features_user = edges.src['user_features']
            features_item = edges.dst['item_features']
        else:
            features_user = edges.dst['user_features']
            features_item = edges.src['item_features']
        '''

        features_u = edges.src['user_features']
        features_v = edges.dst['item_features']
        # print('---')
        # print(features_u)
        # print(features_v)
        # print(features_u.size())
        # print(features_v.size())
        _features = torch.cat([features_u, features_v], 1)
        # print(_features.size())
        # print(_features)
        score = self.W(torch.cat([features_u, features_v], dim = 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.nodes['user'].data['user_features'] = h['user']
            graph.nodes['item'].data['item_features'] = h['item']
            # print('---')
            # print(graph)
            # print(graph.nodes['user'].data['user_features'])
            # print(graph.nodes['item'].data['item_features'])
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

'''
class HeteroGraphSAGE(nn.Module):
    def __init__(self, mods, aggregate = 'sum'):
        super(HeteroGraphSAGE, self).__init__()
        self.mods = nn.ModuleDict(mods)

        if (isinstance(aggregate, str)):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def apply_edges(self, edges):
        pass
    def forward(self, graph, h, etpye):
        pass
'''

print(g.etypes)
import torch.nn.functional as F

import dgl.nn as dglnn
class GraphSAGE(nn.Module):
    # here h_feats_0 is number_user_features, h_feats_1 is number_item_features
    def __init__(self, in_feats, h_feats_0, h_feats_1, rel_names):
        super(GraphSAGE, self).__init__()
        # self.conv1 = GraphSAGE(h_feats_0, h_feats_0, 'mean')
        # self.conv2 = GraphSAGE(h_feats_0, h_feats_0, 'mean')
        # self.conv3 = GraphSAGE(h_feats_1, h_feats_1, 'mean')
        # self.conv4 = GraphSAGE(h_feats_1, h_feats_1, 'mean')
        
        self.pred = MLPPredictor(h_feats_0, h_feats_1, len(rel_names))
        # user
        self.conv0 = dglnn.GraphConv(h_feats_0, h_feats_0)
        # item
        self.conv1 = dglnn.GraphConv(h_feats_1, h_feats_1)

        self.CONV0 = dglnn.HeteroGraphConv({'evaluate': self.conv0, 'evaluated': self.conv1}, aggregate = 'sum')
        self.CONV1 = dglnn.HeteroGraphConv({'evaluate': self.conv1, 'evaluated': self.conv0}, aggregate = 'sum')
        
        # self.conv2 = dglnn.GraphConv(in_feats, h_feats_0)
        # self.conv3 = dglnn.GraphConv(in_feats, h_feats_1)
        self.conv2 = torch.nn.Linear(in_feats, h_feats_0)
        self.conv3 = torch.nn.Linear(in_feats, h_feats_1)

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
            # print(inputs)
            # print(inputs['user'].size())
            # print(inputs['item'].size())
            print('work')
            h = self.CONV0(g, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            # print(h)
            # print(h['user'].size())
            # print(h['item'].size())
            _cat = {'user': torch.cat([inputs['user'], h['user']], dim = 1), 'item': torch.cat([inputs['item'], h['item']], dim = 1)}
            _temp = {}
            _temp['user'] = self.conv2(_cat['user'])
            _temp['item'] = self.conv3(_cat['item'])
            # print(_temp)
            # print(_temp['user'].size())
            '''
            h = self.CONV1(g, h)
            print(h)
            print(h['user'].size())
            print(h['item'].size())
            '''
            dec_graph = g['user', :, 'item']
            # print(dec_graph)
            res = self.pred(dec_graph, _temp)
            # print(res)
            res = torch.softmax(res, dim = 1)
            # print(res)
            return res
            return _temp


model = GraphSAGE(number_user_features + number_item_features, number_user_features, number_item_features, g.etypes)
print(next(model.parameters()).device)
model = model.cuda()
print(next(model.parameters()).device)
user_features = g.nodes['user'].data['features'].to('cuda:0')
item_features = g.nodes['item'].data['features'].to('cuda:0')
# print('----')
# print(user_features.size())
node_features = {'user': user_features, 'item': item_features}
node_features = {key:node_features[key] for key in node_features}
opt = torch.optim.Adam(model.parameters())
epoch = 1000
_labels = [[1 - _, _] for _ in labels]
# print(_labels)
labels_tensor = torch.tensor(_labels)
res = []
# res.to(device = torch.device('cuda:0'))
for i in range(1, epoch + 1): 
    # model(g, {'user': user_features, 'item': item_features})
    res = model(g, node_features)
    dif = (res[labeled] - labels_tensor[labeled])
    
    loss = (dif.mul(dif)).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (i % 5 == 0):
        print("epoch " + str(i) + "/" + str(epoch) + " : loss = ", str(loss))
        _res = res[labeled]
        _res = torch.max(_res, dim = 1)
        _res = _res[1]
        _label = torch.tensor(labels)
        _label = _label[labeled]
        print(_res)
        print(_label)
        # _label = labels[labeled]
        auc = (_res == _label).sum()
        print("auc : " + str(auc.item()) + "/" + str(len(labeled)))
        # exit()
    # print(res)
    # print(labels[0])
    # print(labels[1])
    # print(labels[2])
    # break
print(labels_tensor[labeled])
print(res[labeled])
# h_dict is res
'''
h_user = h_dict['user']
h_item = h_dict['item']

print(h_user)
print()
print(h_item)
'''