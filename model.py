import torch
import dgl

# train_epoch = 0
class MLPPredictor(torch.nn.Module):
    def __init__(self, in_features_0, in_features_1, out_classes):
        super().__init__()
        self.train_epoch = 0
        self.W = torch.nn.Linear(in_features_0 + in_features_1, out_classes)
    def apply_edges(self, edges):
        self.train_epoch = self.train_epoch + 1
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
        if self.train_epoch % 50 == 0:
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


class GraphSAGE(torch.nn.Module):
    # here h_feats_0 is number_user_features, h_feats_1 is number_item_features
    def __init__(self, in_feats, h_feats_0, h_feats_1, rel_names):
        super(GraphSAGE, self).__init__()
        self.train_epoch = 0

        # self.conv1 = GraphSAGE(h_feats_0, h_feats_0, 'mean')
        # self.conv2 = GraphSAGE(h_feats_0, h_feats_0, 'mean')
        # self.conv3 = GraphSAGE(h_feats_1, h_feats_1, 'mean')
        # self.conv4 = GraphSAGE(h_feats_1, h_feats_1, 'mean')
        
        self.pred = MLPPredictor(h_feats_0, h_feats_1, 2)
        # user
        self.conv0 = dgl.nn.GraphConv(h_feats_0, h_feats_0)
        # item
        self.conv1 = dgl.nn.GraphConv(h_feats_1, h_feats_1)

        self.CONV0 = dgl.nn.HeteroGraphConv({'evaluate': self.conv0, 'evaluated': self.conv1}, aggregate = 'sum')
        self.CONV1 = dgl.nn.HeteroGraphConv({'evaluate': self.conv1, 'evaluated': self.conv0}, aggregate = 'sum')
        
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


    def forward(self, g):
        with g.local_scope():
            self.train_epoch = self.train_epoch + 1
            # print(inputs)
            # print(inputs['user'].size())
            # print(inputs['item'].size())
            # h = self.CONV0(g, inputs)
            for k in range(2):
                g.multi_update_all({'evaluate': (dgl.function.copy_u('features', 'm'), dgl.function.mean('m', 'h')),
                                    'evaluated': (dgl.function.copy_u('features', 'm'), dgl.function.mean('m', 'h'))},
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
                if self.train_epoch % 50 == 0:
                    # print('cat ', _cat)
                    # print('_temp ', _temp)
                    # print('_temp max', max(_temp[k].max() for k in _temp))
                    pass
                _temp = {key: _temp[key].cuda() for key in _temp}
                if (k == 0): _temp = {k: torch.nn.functional.relu(v) for k, v in _temp.items()}
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
            # return res
            res = torch.softmax(res, dim = 1)
            # print(res)
            return res
