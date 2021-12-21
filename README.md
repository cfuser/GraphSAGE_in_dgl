This is [**GraphSAGE**](https://arxiv.org/abs/1706.02216) within **DGL**.

The paper: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

**GraphSAGE** is an algorithm that aggregate the features of neighbor nodes and self nodes simultaneously without considering the order of nodes. It requires that the features of nodes should be same. However, it doesn't work well in heterogeneous graph, especially in the case that the features of nodes are different in dimensions.

We change some details.

1.  Firstly, we aggregate the features of neighbor nodes, then concatenate with self nodes. Next we use an MLP to get features of the same size as self nodes. It's called one layer of **new GraphSAGE**. We have two **new GraphSAGE** in our model.
2.  In paper, **GraphSAGE** is used to node classification and supervised. While our target is to link classification and semi-supervised. For former problem, we concatenate the features of nodes with unidirectional edge, and use an MLP to a two classification problem. For latter problem, we temporarily use supervised loss function. **Work to do**


result:
test.py without gpu
1 min 10+ epoch

_cuda.py
1 min 100+ epoch

test2.py to evaluate

初赛 auc 0.848
supervised

复赛 0.99527295125
semi-supervised

code learning:

1. Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks

Work To Do:

1. change the function of aggregation. In the code, we use a GraphConv layer, however, it should be delivered in message

---
code link: https://github.com/cfuser/GraphSAGE_in_dgl

Wed Dec 15 11:57:56 2021

Pseudo Code

```python
load train.csv
class model
train
load predict.csv
print(accuracy or other evaluations)
```



---

Wed Dec 15 12:02:33 2021

Work To Do

1.  semi-supervised loss function 

2.  finish test-set loading

   2.1 first, load like training-set

   2.2 forward() to aggregate neighbors, predict() to get result 

   2.3 adjust model

3.  

---

Bug List:

1. multiedge in preliminary contest train set, like 46 98, which should be removed
2. the label sequence and train sequence is not identical
3. loss function and jump to nan, it's in the process of self.W, need more information

Unknown:

1. expand_as_pair, how it works? I have known that it returns two input features, but why we need it?
