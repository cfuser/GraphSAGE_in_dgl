
import torch
 
label = torch.tensor([[1.0, 0.0, 0.0],
                      [3.0, 0.0, 1.0]])
print((label == 0).nonzero())
