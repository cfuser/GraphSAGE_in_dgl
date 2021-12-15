import csv
import torch
import dgl

from dgl.data import dgl_dataset

f = open('D:\subject\graduate\graph computation\data\offline\\train0.csv', 'r')
reader = csv.reader(f)

'''
day = 0
num = 0
dic = [{}, {}]

for i in reader:
    # print(i)
    if (int(i[1]) > day):
        dic = [{}, {}]
        day = int(i[1])
    if (not dic[0].get(i[3])):
        dic[0][i[3]] = num
        dic[1][i[3]] = i[2]
    else:
        if (dic[1][i[3]] != i[2]):
            print(dic[0].get(i[3]))
            print(num)
            print()
            break
    num = num + 1
'''

'''
str1 = '8.10504 8.1322 10.30608 7.96097 11.30549 5.32912 2.58496 7.83289 7.53916 25.5759 8.20457 2.67482 2.91158 2.59396 8.06609 10.8009 14.42581 13.90407 2.58496 3.14405 2.67811 3.16993 2.56071 3.16993 10.28983 7.94837 1.0 8.16491 7.8009 2.90589 10.4947 8.0 8.10852 10.53337 7.83289 7.97154 8.10852 1.0 10.53337 7.83289 2.32193 2.58496 8.01681 2.80735 0.18057 0.26303 7.21594 0.41504 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 0.03626 0.04382 3.62838 3.51937 0.05857 0.0596 8.06609 1.0 8.03746 1.0 8.55842 8.53138 0.01708 0.02407 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.58496 3.3799 0.0 0.0 0.0 0.0 5.49841 1.58496 1.58496 1.58496 1.0 1.0 3.3799 0.0 3.3799 1.0 5.2854 1.0 5.49185 4.58496 6.48236 3.3799 1.0 1.0 3.3799 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 6.48236 1.58496 1.58496 1.58496 5.49841 1.58496 0.0 0.0 0.34104 0.41504 3.51618 0.26303 0.26303 0.18057 0.41504 0.48543 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0'
str2 = '8.10504 8.1322 10.30608 7.96097 11.30549 5.32912 2.58496 7.83289 7.53916 25.5759 8.20457 2.67482 2.91158 2.59396 8.06609 10.8009 14.42581 13.90407 2.58496 3.14405 2.67811 3.16993 2.56071 3.16993 10.28983 7.94837 1.0 8.16491 7.8009 2.90589 10.4947 8.0 8.10852 10.53337 7.83289 7.97154 8.10852 1.0 10.53337 7.83289 2.32193 2.58496 8.01681 2.80735 0.18057 0.26303 7.21594 0.41504 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 0.03626 0.04382 3.62838 3.51937 0.05857 0.0596 8.06609 1.0 8.03746 1.0 8.55842 8.53138 0.01708 0.02407 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.58496 3.70044 7.27612 0.0 0.0 0.0 0.0 6.39297 4.64386 5.24793 4.08746 1.58496 2.32193 6.86419 8.92926 7.27612 2.0 5.04439 1.0 8.49586 7.22882 10.96138 8.85487 1.0 2.32193 8.85487 1.0 0.0 2.32193 1.0 6.93811 2.32193 1.0 2.0 0.0 0.0 10.55786 3.58496 5.08746 4.0 6.254 4.39232 0.0 0.0 0.95109 1.1375 6.37353 0.58496 0.70782 0.34104 2.19114 2.65535 0.0 0.55254 0.0 0.0 0.0 0.0 0.0 0.0 0.09311 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0'
for i in range(len(str1)):
    if (str1[i] != str2[i]):
        print(i)
        print(str1[:i + 1])
        print(str2[:i + 1])
        break
    pass
'''

'''
str3 = '8.10504 8.1322 10.30608 7.96097 11.30549 5.32912 2.58496 7.83289 7.53916 25.5759 8.20457 2.67482 2.91158 2.59396 8.06609 10.8009 14.42581 13.90407 2.58496 3.14405 2.67811 3.16993 2.56071 3.16993 10.28983 7.94837 1.0 8.16491 7.8009 2.90589 10.4947 8.0 8.10852 10.53337 7.83289 7.97154 8.10852 1.0 10.53337 7.83289 2.32193 2.58496 8.01681 2.80735 0.18057 0.26303 7.21594 0.41504 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 0.03626 0.04382 3.62838 3.51937 0.05857 0.0596 8.06609 1.0 8.03746 1.0 8.55842 8.53138 0.01708 0.02407 1.0 0.0 0.0 0.0 0.0 0.0 0.0'
print(str3)
print(str3.count(' ') + 1)
'''

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
                ('item', 'evaluted', 'user'): (link_tensor[1], link_tensor[0])
                }
g = dgl.heterograph(graph_data)
print(g)


for i in record:
    features.append(i['features'])
    # item_features.append(i['features'][:M])
    # user_features.append(i['features'][M:])
    labels.append(i['label'])

#print(g)
print(torch.tensor(user_features).size())
print(torch.tensor(item_features).size())
g.nodes['user'].data['features'] = torch.tensor(user_features)
g.nodes['item'].data['features'] = torch.tensor(item_features)

#g.ndata['features'] = torch.tensor(item_features)
#g.edata['features'] = torch.tensor(labels)

print(g)
# g = g.to('cuda:0')
import torch.nn as nn
import dgl.function as fn
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h   #一次性为所有节点类型的 'h'赋值
            graph.apply_edges(fn.u_dot_v('features', 'features', 'score'), etype=etype)
            return graph.edges[etype].data['score']

class MLPPredictor(nn.Module):
    def __init__(self, in_features_0, in_features_1, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features_0 + in_features_1, out_classes)
    def apply_edges(self, edges):
        features_u = edges.src['h']
        features_v = edges.dst['h']
        score = self.W(torch.cat(features_u, features_v), 1)
        return {'score': score}
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class HeteroGraphSAGE(nn.Module):
    def __init__(self, mods, aggregate = 'sum'):
        super(HeteroGraphSAGE, self).__init__()
        self.mods = nn.ModuleDict(mods)
        '''
        if (isinstance(aggregate, str)):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate
        '''
    def apply_edges(self, edges):
        pass
    def forward(self, graph, h, etpye):
        pass

print(g.etypes)
import torch.nn.functional as F

### train
'''
epoch = 100
for e in range(epoch):
    logits = model(graph_model, features)
    pred = logits.argmax(1)
    loss = F.cross_entropy(logits, labels)
    for i, j in logits:
        loss = loss + (i - j) ** 2
    
    pass
'''

# unnecessary process
'''
print(link_tensor.size())
g = dgl.graph((link_tensor[0], link_tensor[1]))
#g = dgl.add_reverse_edges(g)
features = []
user_features = []
item_features = []
labels = []
for i in record:
    features.append(i['features'])
    item_features.append(i['features'][:M])
    user_features.append(i['features'][M:])
    labels.append(i['label'])
print(g)
print(torch.tensor(item_features).size())
g.ndata['features'] = torch.tensor(item_features)
g.edata['features'] = torch.tensor(labels)
'''