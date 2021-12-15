import dgl
import torch

g = dgl.DGLGraph()

g.add_nodes(10)
g.add_edges((1, 2, 3, 4, 5, 6, 7, 8, 9, 0), (7, 3, 4, 6, 5, 2, 1, 0, 8, 9))

def test(g):
    g.edata['h1'] = torch.ones((g.num_edges(), 3))
    with g.local_scope():
        g.edata['h'] = torch.ones((g.num_edges(), 3))
        g.edata['h2'] = torch.ones((g.num_edges(), 3))
        return g.edata['h']
    
d = test(g)
print(d)
print(g)
print(g.device)
print(torch.cuda.is_available())
devices = ['cuda' if torch.cuda.is_available() else 'cpu']
print(devices)
cuda_g = g.to('cuda:0')
print(cuda_g)

tensor = [1, 2, 3]