from typing import Optional, Tuple

import torch

from .utils import broadcast

# @torch.jit.script
# def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#                  out: Optional[torch.Tensor] = None,
#                  dim_size: Optional[int] = None) -> torch.Tensor:
#
#     out = scatter_sum(src, index, dim, out, dim_size)
#     dim_size = out.size(dim)
#
#     index_dim = dim
#     if index_dim < 0:
#         index_dim = index_dim + src.dim()
#     if index.dim() <= index_dim:
#         index_dim = index.dim() - 1
#
#     ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
#     count = scatter_sum(ones, index, index_dim, None, dim_size)
#     count.clamp_(1)
#     count = broadcast(count, out, dim)
#     out.div_(count)
#     return out
#
#
# @torch.jit.script
# def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#                 out: Optional[torch.Tensor] = None,
#                 dim_size: Optional[int] = None
#                 ) -> Tuple[torch.Tensor, torch.Tensor]:
#     return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)
#
#
# @torch.jit.script
# def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#                 out: Optional[torch.Tensor] = None,
#                 dim_size: Optional[int] = None
#                 ) -> Tuple[torch.Tensor, torch.Tensor]:
#     return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = src.size()
        if dim_size is not None:
            size[dim] = dim_size  # 被压缩的维度
        elif index.numel() == 0:
            size[dim] = 0  # 如果index为空向量，则被压缩维度变为0维（即为空）
        else:
            size[dim] = int(index.max()) + 1  # 默认压缩值值index.max()+1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


@torch.jit.script
def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:

    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    # elif reduce == 'mean':
    #     return scatter_mean(src, index, dim, out, dim_size)
    # elif reduce == 'min':
    #     return scatter_min(src, index, dim, out, dim_size)[0]
    # elif reduce == 'max':
    #     return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError


