from .loop import (contains_self_loops, remove_self_loops,
                   segregate_self_loops, add_self_loops,
                   add_remaining_self_loops)
from .isolated import contains_isolated_nodes, remove_isolated_nodes
from .undirected import is_undirected, to_undirected

from .scatter import scatter_
__all__ =[
    'scatter_',
    'add_self_loops',
    'add_remaining_self_loops',
    'remove_self_loops',
    'contains_self_loops',
    'segregate_self_loops',
    'contains_isolated_nodes',
    'remove_isolated_nodes',
    'is_undirected',
    'to_undirected'
]