import model
import dgl
import torch
import sklearn.metrics
import numpy

# graph load and model defintion
path = 'D:/subject/graduate/graph computation/data/offline/predata/'
graph = dgl.load_graphs(path + 'data.bin')
# print(graph)
number_user_features = graph[0][0].nodes['user'].data['features'].size()[1]
number_item_features = graph[0][0].nodes['item'].data['features'].size()[1]
graph[0][0] = graph[0][0].to('cuda:0')
labels = graph[0][0].edges['evaluate'].data['label']
labels = labels.float()
labeled = (labels != -1).nonzero()
labeled = labeled.squeeze(1)
unlabeled = (labels == -1).nonzero()
unlabeled = unlabeled.squeeze(1)
# _labels = 1 - labels
# print(_labels)
# print(labeled)
# print(unlabeled)

model = model.GraphSAGE(number_user_features + number_item_features, number_user_features, number_item_features, 2)
model.to('cuda:0')

# train

opt = torch.optim.Adam(model.parameters())
print('input number of epoch: ')
epoch = int(input())
for i in range(1, epoch + 1):
    res = model(graph[0][0])
    res = res.squeeze(1)
    preds = res[:, 1]
    if (i % 50 == 0):
        print('epoch ' + str(i) + '/' + str(epoch) + ' : res = ')
        print(res)
    loss = torch.nn.functional.binary_cross_entropy(res[:, 1], labels[labeled])
    if (len(unlabeled)):
        loss = loss + torch.nn.functional.binary_corss_entropy(res[:, 1], res[:, 1])
    
    _res = res[labeled]
    # print(_res)
    _res_st = torch.max(_res, dim = 1)
    _res_value = _res_st[0]
    _res_indix = _res_st[1]
    _label = labels
    _label = _label[labeled]

    if (i % 50 == 0):
        print("epoch " + str(i) + "/" + str(epoch) + " : loss = ", str(loss))
        # print(res)
        # print(_res)
        # print(_res_st)
        print(_res_indix)
        print(_label)
        # _label = labels[labeled]
        acc = (_res_indix.to('cuda:0') == _label.to('cuda:0')).sum()
        print("acc : " + str(acc.item()) + "/" + str(len(labeled)))
        y = _label.to('cpu')
        preds = numpy.array(res.detach().cpu()[:, 1])
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, preds, pos_label = 1)
        auc = sklearn.metrics.roc_auc_score(y, preds)
        print('auc:', auc)
        print()
        # exit()

    opt.zero_grad()
    loss.backward()
    opt.step()

torch.save(model, './model.pkl')