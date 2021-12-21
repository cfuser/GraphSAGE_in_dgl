import csv
from dgl.batch import unbatch
import torch
import dgl
import numpy as np
from dgl.data import dgl_dataset
import sklearn.metrics

path = 'D:\subject\graduate\graph computation\data\offline\predata\\'
# f_train = open('D:\subject\graduate\graph computation\data\offline\\train0.csv', 'r')
f_train = open(path + 'train.csv', 'r')
reader = csv.reader(f_train)

M = 32 # 79
N = 75 # 152

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
unlabeled = []
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
    else:
        unlabeled.append(_edge)
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
g = g.to('cuda:0')
print(g)


number_user_features = N - M
number_item_features = N - number_user_features
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
        if i % 50 == 0:
            print(_features)
            print(score)
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.nodes['user'].data['user_features'] = h['user']
            graph.nodes['item'].data['item_features'] = h['item']
            '''
            if i % 50 == 0:
                print('---')
                print(graph.edges())
                print()
            '''
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
        
        self.pred = MLPPredictor(h_feats_0, h_feats_1, 2)
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
        self.conv4 = torch.nn.Linear(in_feats, h_feats_0)
        self.conv5 = torch.nn.Linear(in_feats, h_feats_1)
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
            # h = self.CONV0(g, inputs)
            for k in range(2):
                g.multi_update_all({'evaluate': (fn.copy_u('features', 'm'), fn.mean('m', 'h')),
                                    'evaluated': (fn.copy_u('features', 'm'), fn.mean('m', 'h'))},
                                    'sum')
                h = g.ndata['h']
                # print(g.nodes['user'].data['h'].size())
                # print(g.nodes['item'].data['h'].size())
                '''
                if i % 50 == 0:
                    print('h:\n', h)
                h = {k: F.relu(v) for k, v in h.items()}
                '''
                # print(h)
                # print(h['user'].size())
                # print(h['item'].size())
                _cat = {'user': torch.cat([g.nodes['user'].data['features'], h['user']], dim = 1), 'item': torch.cat([g.nodes['item'].data['features'], h['item']], dim = 1)}
                _temp = {}
                if k == 0:
                    _temp['user'] = self.conv2(_cat['user'])
                    _temp['item'] = self.conv3(_cat['item'])
                else:
                    _temp['user'] = self.conv4(_cat['user'])
                    _temp['item'] = self.conv5(_cat['item'])
                if i % 50 == 0:
                    print('cat ', _cat)
                    print('_temp ', _temp)
                # _temp = {key: _temp[key].cuda() for key in _temp}
                # if (k == 0): _temp = {k: F.relu(v) for k, v in _temp.items()}
                g.ndata['features'] = _temp
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
            return res
            res = torch.softmax(res, dim = 1)
            # print(res)
            return res
            return _temp


model = GraphSAGE(number_user_features + number_item_features, number_user_features, number_item_features, 2)
model.cuda()
user_features = g.nodes['user'].data['features'].to('cuda:0')
item_features = g.nodes['item'].data['features'].to('cuda:0')
# print('----')
# print(user_features.size())
node_features = {'user': user_features, 'item': item_features}
node_features = {key:node_features[key].cuda() for key in node_features}
opt = torch.optim.Adam(model.parameters())
print('input number of epoch: ')
epoch = int(input())
# epoch = 5

_labels = [[1 - _, _] for _ in labels]
# print(_labels)
labels_tensor = torch.tensor(_labels)
res = []
i = 0
for i in range(1, epoch + 1):
    res = model(g, node_features)
    if (i % 50 == 0):
        print('epoch ' + str(i) + '/' + str(epoch) + ' : res = ')
        print(res)
    res = torch.softmax(res, dim = 1)
    dif = (res[labeled].to('cuda:0') - labels_tensor[labeled].to('cuda:0'))
    
    _res = res[labeled]
    # print(_res)
    _res_st = torch.max(_res, dim = 1)
    _res_value = _res_st[0]
    _res_indix = _res_st[1]
    _label = torch.tensor(labels)
    _label = _label[labeled]
    # print(_res.shape[0])
    loss = - (torch.log(_res).to('cuda:0') * labels_tensor[labeled].to('cuda:0')).sum() / _res.shape[0]
    if (len(unlabeled)):
        loss = loss - (torch.log(res[unlabeled]) * res[unlabeled]).sum() / len(unlabeled)
    # loss = - ((torch.log(_res) * labels_tensor[labeled]).sum() + (torch.log(res[unlabeled]) * res[unlabeled]).sum()) / _edge
    # print(loss)
    # loss = (dif.mul(dif)).mean()
    
    if (i % 50 == 0):
        print("epoch " + str(i) + "/" + str(epoch) + " : loss = ", str(loss))
        
        print(_res_indix)
        print(_label)
        # _label = labels[labeled]
        acc = (_res_indix.to('cuda:0') == _label.to('cuda:0')).sum()
        print("acc : " + str(acc.item()) + "/" + str(len(labeled)))
        y = _label
        preds = np.array(res.detach().cpu()[:, 1])
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, preds, pos_label = 1)
        auc = sklearn.metrics.roc_auc_score(y, preds)
        print('auc:', auc)
        print()
        # exit()
    
    opt.zero_grad()
    loss.backward()
    opt.step()
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

# predict_0.csv
pass
# f_predict = open('D:\subject\graduate\graph computation\data\offline\\predict0.csv', 'r')
f_predict = open(path + 'predict.csv', 'r')
predict_reader = csv.reader(f_predict)

predict_record = []
predict_node_num = 0
predict_user = {}
predict_item = {}

predict_features = []
predict_user_features = []
predict_item_features = []
predict_labels = []
predict_uuids = []
for i in predict_reader:
    temp = {}
    temp['uuid'] = int(i[0])
    predict_uuids.append(int(i[0]))
    temp['visit_time'] = int(i[1])
    temp['user_id'] = int(i[2])
    temp['item_id'] = int(i[3])
    features = i[4].split()
    features = [float(_) for _ in features]
    temp['features'] = features
    # temp['label'] = int(i[5])
    predict_record.append(temp)
    if not(int(i[2]) in predict_user):
        predict_user[int(i[2])] = predict_node_num
        predict_node_num = predict_node_num + 1
        predict_user_features.append(features[M:])
    else:
        # detect whether the feature of user if same or not
        pass

predict_uuids = torch.tensor(predict_uuids)
predict_user_idx_begin = 0
predict_item_idx_begin = predict_node_num

predict_link = [[], []]
for i in predict_record:
    if (not(i['item_id']) in predict_item):
        predict_item[i['item_id']] = predict_node_num
        predict_node_num = predict_node_num + 1
        predict_item_features.append(i['features'][:M])
    else:
        # detect whether the features of item is same or not
        pass
    
    predict_link[0].append(predict_user[i['user_id']])
    predict_link[1].append(predict_item[i['item_id']] - predict_item_idx_begin)

predict_link_tensor = torch.tensor(predict_link)

predict_graph_data = {('user', 'evaluate', 'item'): (predict_link_tensor[0], predict_link_tensor[1]),
                        ('item', 'evaluated', 'user'): (predict_link_tensor[1], predict_link_tensor[0])
                        }

predict_g = dgl.heterograph(predict_graph_data)
print(predict_g)

# process data to normalize
number_user_features = N - M
number_item_features = N - number_user_features
# print(torch.tensor(user_features))
predict_min = torch.min(torch.tensor(predict_user_features), dim = 0)
predict_max = torch.max(torch.tensor(predict_user_features), dim = 0)
# print(_min)
# print(_max)
predict_dif = predict_max[0] - predict_min[0]
predict_nonzero_idx = torch.nonzero(predict_dif == 0)

# torch.set_printoptions(precision=8)

# print(_dif)
# print(_nonzero_idx)
predict_dif[predict_nonzero_idx] = 1
# print(_dif)
predict_temp = (torch.tensor(predict_user_features) - predict_min[0]) / predict_dif
predict_user_features = predict_temp.numpy().tolist()
# print((torch.tensor(user_features) - _min[0]) / _dif)
# _max = torch.max(_temp, dim = 0)
# print(_max)
predict_min = torch.min(torch.tensor(predict_item_features), dim = 0)
predict_max = torch.max(torch.tensor(predict_item_features), dim = 0)
predict_dif = predict_max[0] - predict_min[0]
predict_nonzero_idx = torch.nonzero(predict_dif == 0)
predict_dif[predict_nonzero_idx] = 1
predict_temp = (torch.tensor(predict_item_features) - predict_min[0]) / predict_dif
predict_item_features = predict_temp.numpy().tolist()


print(torch.tensor(predict_user_features).size())
print(torch.tensor(predict_item_features).size())
predict_g.nodes['user'].data['features'] = torch.tensor(predict_user_features)
predict_g.nodes['item'].data['features'] = torch.tensor(predict_item_features)
print(predict_g)
print(predict_g.nodes['user'].data['features'])
print(predict_g.etypes)

predict_user_features = predict_g.nodes['user'].data['features'].to('cuda:0')
predict_item_features = predict_g.nodes['item'].data['features'].to('cuda:0')

predict_node_features = {'user': predict_user_features, 'item': predict_item_features}

predict_g = predict_g.to('cuda:0')
i = 0
predict_res = model(predict_g, predict_node_features)
print(predict_res)
print(predict_res.shape)

predict_res_st = torch.max(predict_res, dim = 1)
predict_res_value = predict_res_st[0]
predict_res_indix = predict_res_st[1]

# truth.csv
# f_truth = open('D:\subject\graduate\graph computation\data\offline\\truth.csv', 'r')
f_truth = open(path + 'truth.csv', 'r')
truth_reader = csv.reader(f_truth)
truth_uuids = []
truth_labels = []
for i in truth_reader:
    truth_uuids.append(int(i[0]))
    truth_labels.append(int(i[1]))

truth_uuids = torch.tensor(truth_uuids)
truth_labels = torch.tensor(truth_labels)

file_handle = open('predict_res.txt', 'w')
x = predict_res_indix.cpu().numpy()
x = x.tolist()
strNums = [str(_) for _ in x]
str1 = "\n".join(strNums)
file_handle.write(str1)
file_handle.close()

file_handle = open('predict_value.txt', 'w')
strNums = [str(_[1].detach().item()) for _ in predict_res]
str2 = "\n".join(strNums)
file_handle.write(str2)
file_handle.close()

_truth_labels = [truth_labels[_ - 1] for _ in predict_uuids]
_truth_labels = torch.tensor(_truth_labels)
predict_acc = (_truth_labels.to('cuda:0') == predict_res_indix.to('cuda:0')).sum()
print("predict acc : " + str(predict_acc.item()) + "/" + str(len(predict_record)))

y = np.array(truth_labels)
preds = np.array(predict_res.detach().cpu()[:, 1])
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, preds, pos_label = 1)
auc = sklearn.metrics.roc_auc_score(y, preds)
print('auc:', auc)
