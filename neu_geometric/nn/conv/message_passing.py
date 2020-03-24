import inspect
from collections import OrderedDict

import torch
from neu_geometric.utils import scatter_

# python中的set,在{}中用逗号隔开
msg_special_args = {'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'}

aggr_special_args = {'index', 'dim_size'}

update_special_args = set([])


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`0`)
    """
    def __init__(self, aggr='add', flow='source_to_target', node_dim=0):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim  # node_dim代表的是图节点所在的维度,一般为0维
        assert self.node_dim >= 0
        # inspect.signature(self.message).parameters操作都是在获取用户自定义函数的参数列表
        # 注意这里self.message调用的是子类的message
        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        # 使用popitem(last=True)让我们按照LIFO(先进后出)顺序删除OrderedDict中的key-value，即删除最后插入的键值对
        # 如果last=False就按照FIFO删除dict中key-value,，即删除最早插入的键值对(我猜测是self)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        # 去掉特殊关键字参数
        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        # 所有三个函数的参数集合
        self.__args__ = set().union(msg_args, aggr_args, update_args)

    # self.__set_size__(size, idx, data)
    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)  # 推断出图中节点数量
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (f'Encountered node tensor with size '
                 f'{tensor.size(self.node_dim)} in dimension {self.node_dim}, '
                 f'but expected size {size[index]}.'))

    def __collect__(self, edge_index, size, kwargs):
        # return self.propagate(edge_index, x=x, norm=norm)
        # kwargs = self.__collect__(edge_index, size, kwargs)
        # 注意默认 i是1, j是0
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        # 注意__args__中不会出现关键字参数
        for arg in self.__args__:
            # 如果参数最后两位不是以_i,_j结尾的
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                # “x_j"参数进入下面的分支
                idx = ij[arg[-2:]]  # idx==0
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)  # data是节点特征矩阵x

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue
                #  edge_index (LongTensor, optional): Graph connectivity in COO format
                #  with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        # size[0] = size[1] 默认是一个正方形矩阵，shape维(node_num, node_num)
        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        # 注意这里及其易错！！！！！！！
        out['edge_index'] = edge_index  # 边的coo格式的tensor矩阵
        out['edge_index_i'] = edge_index[i]  # 边的终点index！！！！
        out['edge_index_j'] = edge_index[j]  # 边的源点index！！！！
        out['size'] = size  # 邻接矩阵的大小
        out['size_i'] = size[i]
        out['size_j'] = size[j]

        # Add special aggregate arguments.
        out['index'] = out['edge_index_i']  # 边的源点index
        out['dim_size'] = out['size_i']

        return out

    # kwargs在GCN中主要包含的是x和norm
    # self.__msg_params__, kwargs
    # def message(self, x_j, norm):
    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size will be
                automatically inferred and assumed to be quadratic.
                (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2

        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        out = self.message(**msg_kwargs)

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update(out, **update_kwargs)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages to node :math:`i` in analogy to
        :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and
        :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        By default, delegates call to scatter functions that support
        "add", "mean" and "max" operations specified in :meth:`__init__` by
        the :obj:`aggr` argument.
        """
        """
        self.aggr == "add"
        inputs == out
        
        """
        # 注意index是列方向上的index（即终点列表）
        return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

    def update(self, inputs):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """

        return inputs
