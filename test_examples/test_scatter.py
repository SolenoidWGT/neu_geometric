import torch
import os
import importlib
import os.path as osp
from torch_scatter import scatter_max
torch.ops.load_library("../my_scatter/_scatter.so")

print(torch.ops.my_scatter.scatter_sum)
print(scatter_max)