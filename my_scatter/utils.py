import torch


@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """

    :param src: index, 假设index为[0,1,2,3,4,5,6]
    :param other: src, 假设src为二维矩阵形式
    :param dim: dim, 假设dim为0
    :return:
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:  # 如果index只有一个纬度
        for _ in range(0, dim):  # 不执行
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):  # for 1 to 2:
        src = src.unsqueeze(-1)  # index变为[[0],[1],[0],[0],[2],[3],[4]]
    src = src.expand_as(other)  # index变为[[0,0,...,0] ,[0,0,...,0],...], 维度和src一致
    return src
