import dgl
import csv
from dgl.data.graph_serialize import save_graphs
import torch

'''
preprocess is the function to process data and store before train, like the name describes
M is the number of item_features, N is the number of all features
user and item is dict to store name and key,
node_num is used to adjust the begin of item when linking,
features is the all features of a record, [0, M) for item, and remaining for user
user_features and item_features is to devide,
labels is the label of record
labeled is the list of record which has been labeled,
unlabeled is opposite
_edge is key of labeled and unlabeled
'''

def preprocess(M, N, process_file):
    graph = []
    for file_name in process_file:
        file = open(file_name, 'r')
        reader = csv.reader(file)
        record = []
        node_num = 0
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
            if len(i) == 6:
                temp['label'] = int(i[5])
                labels.append(int(i[5]))
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
                ('item', 'evaluated', 'user'): (link_tensor[1], link_tensor[0])
                }
        g = dgl.heterograph(graph_data)
        if (labels):
            g.edges['evaluate'].data['label'] = torch.tensor(labels)
            g.edges['evaluated'].data['label'] = torch.tensor(labels)
        number_user_features = N - M
        number_item_features = N - number_user_features
        user_features = torch.tensor(user_features)
        item_features = torch.tensor(item_features)
        user_features = torch.nn.functional.normalize(user_features, p = 1, dim = 1)
        item_features = torch.nn.functional.normalize(item_features, p = 1, dim = 1)
        # print(g)
        # print(torch.tensor(user_features).size())
        # print(torch.tensor(item_features).size())
        g.nodes['user'].data['features'] = user_features.clone().detach()
        g.nodes['item'].data['features'] = item_features.clone().detach()
        graph.append(g)
    return graph

path = 'D:/subject/graduate/graph computation/data/offline/predata/'

M = 32 # 79
N = 75 # 152
process_file = [path + 'train.csv', path + 'predict.csv']
graph = preprocess(M, N, process_file)
graph_labels = {'train.csv': torch.tensor([0]), 'predict.csv': torch.tensor([1])}
save_graphs(path + 'data.bin', graph, graph_labels)

