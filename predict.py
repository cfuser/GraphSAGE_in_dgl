import numpy
import torch
import dgl
import sklearn.metrics
import csv
import model

path = 'D:/subject/graduate/graph computation/data/offline/predata/'
graph = dgl.load_graphs(path + 'data.bin')
# print(graph)
number_user_features = graph[0][1].nodes['user'].data['features'].size()[1]
number_item_features = graph[0][1].nodes['item'].data['features'].size()[1]
graph[0][1] = graph[0][1].to('cuda:0')

labels = []
f_truth = open(path + 'truth.csv', 'r')
reader = csv.reader(f_truth)
for i in reader:
    labels.append(int(i[1]))
labels = torch.tensor(labels)
labels = labels.float()
labels = labels.to('cuda:0')

model = model.GraphSAGE(number_user_features + number_item_features, number_user_features, number_item_features, 2)
model.load_state_dict(torch.load('./model.pkl'))
model.cuda()

res = model(graph[0][1])
res = res.squeeze(1)
preds = res[:, 1]
print(res)
loss = torch.nn.functional.binary_cross_entropy(res[:, 1], labels)
_res = res
# print(_res)
_res_st = torch.max(_res, dim = 1)
_res_value = _res_st[0]
_res_indix = _res_st[1]
_label = labels
_label = _label

print("loss = ", str(loss))
# print(res)
# print(_res)
# print(_res_st)
print(_res_indix)
print(_label)
# _label = labels[labeled]
acc = (_res_indix.to('cuda:0') == _label.to('cuda:0')).sum()
print("acc : " + str(acc.item()) + "/" + str(len(labels)))
y = _label.to('cpu')
preds = numpy.array(res.detach().cpu()[:, 1])
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, preds, pos_label = 1)
auc = sklearn.metrics.roc_auc_score(y, preds)
print('auc:', auc)
print()
# exit()