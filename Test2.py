import csv
import torch
import sklearn.metrics
import numpy as np

f_train = open('D:\subject\graduate\graph computation\data\offline\\truth.csv', 'r')
reader = csv.reader(f_train)


node_num = 0
same_node_num = 0
y = []
preds = []

for i in reader:
    if (i[1] == i[2]):
        same_node_num = same_node_num + 1
    node_num = node_num + 1
    y.append(int(i[1]))
    preds.append(float(i[3]))
    if node_num == 50000:
        break

print(same_node_num)
print(node_num)

y = np.array(y)
preds = np.array(preds)
print(y)
print(preds)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, preds, pos_label = 1)

print(fpr)
print(tpr)
print(thresholds)
auc = sklearn.metrics.roc_auc_score(y, preds)
print(auc)