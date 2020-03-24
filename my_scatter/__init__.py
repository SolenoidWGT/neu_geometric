import os
import importlib
import os.path as osp
import torch

expected_torch_version = (1, 4)
for library in ['_scatter']:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        library, [osp.dirname(__file__)]).origin)

# try:
#     # for library in ['_version', '_scatter', '_segment_csr', '_segment_coo']:
#     #     torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
#     #         library, [osp.dirname(__file__)]).origin)
#     torch.ops.load_library("_scatter.so")
# except OSError as e:
#     major, minor = [int(x) for x in torch.__version__.split('.')[:2]]
#     t_major, t_minor = expected_torch_version
#     if major != t_major or (major == t_major and minor != t_minor):
#         raise RuntimeError(
#             f'Expected PyTorch version {t_major}.{t_minor} but found '
#             f'version {major}.{minor}.')
#     raise OSError(e)
# except AttributeError as e:
#     print("进入第二except!!，请进行检查")
#     if os.getenv('BUILD_DOCS', '0') != '1':
#         raise AttributeError(e)
#
# # 进行cuda版本检查
# if torch.version.cuda is not None:  # pragma: no cover
#     cuda_version = torch.ops.torch_scatter.cuda_version()
#
#     if cuda_version == -1:
#         major = minor = 0
#     elif cuda_version < 10000:
#         major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
#     else:
#         major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
#     t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]
#
#     if t_major != major or t_minor != minor:
#         raise RuntimeError(
#             f'Detected that PyTorch and torch_scatter were compiled with '
#             f'different CUDA versions. PyTorch has CUDA version '
#             f'{t_major}.{t_minor} and torch_scatter has CUDA version '
#             f'{major}.{minor}. Please reinstall the torch_scatter that '
#             f'matches your PyTorch install.')
#

from .scatter import scatter
from .scatter import scatter_add
__all__ = [
    'scatter',
    'scatter_add'
]
