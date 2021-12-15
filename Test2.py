import torch
from torch._C import FloatType
import torch.nn.functional as F
pred = torch.tensor([[1, 1], [3, 3], [2, 2]], dtype=torch.float)
targ = torch.tensor([[0, 0], [0, 0], [2, 2]], dtype=torch.float)
print(F.mse_loss(pred, targ))