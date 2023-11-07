import numpy as np

from tensor.utils.dimtree import dimtree


class HTucker:
    # Imported instance methods
    from ._rebuild import full, rebuild_tensor_helper
    from ._size import get_size
    from ._getitem import get

    # Imported static methods
    from ._trunc_rank import trunc_rank
    trunc_rank = staticmethod(trunc_rank)

    # Imported class methods
    from ._truncate import truncate
    from ._orthogonalize import orthogonalize
    from ._add import add
    from ._inner_product import inner_product
    from ._mode_multiplication import mode_multiplication
    from ._ews_mode_multiplication import ews_mode_multiplication
    from ._norm import norm
    from ._gramians_orthog import gramians_orthog
    from ._contraction import contract
    from ._change_root import change_root
    from ._change_dimtree import change_dimtree
    from ._squeeze import squeeze
    from ._ews_multiplication import ews_multiplication
    from ._truncate_htucker import truncate_htucker
    from ._ews_mode_multiplication import ews_mode_multiplication
    from ._scalar_multiplication import scalar_mul
    from ._add_and_truncate import add_and_truncate
    from ._gramians_sum import gramians_sum
    from ._gramians_nonorthog import gramians_nonorthog
    from ._concat import concat
    truncate = classmethod(truncate)
    orthogonalize = classmethod(orthogonalize)
    add = classmethod(add)
    inner_product = classmethod(inner_product)
    mode_multiplication = classmethod(mode_multiplication)
    ews_mode_multiplication = classmethod(ews_mode_multiplication)
    norm = classmethod(norm)
    gramians_orthog = classmethod(gramians_orthog)
    contract = classmethod(contract)
    change_root = classmethod(change_root)
    change_dimtree = classmethod(change_dimtree)
    squeeze = classmethod(squeeze)
    ews_multiplication = classmethod(ews_multiplication)
    trunc_rank = classmethod(trunc_rank)
    truncate_htucker = classmethod(truncate_htucker)
    scalar_mul = classmethod(scalar_mul)
    add_and_truncate = classmethod(add_and_truncate)
    gramians_sum = classmethod(gramians_sum)
    gramians_nonorthog = classmethod(gramians_nonorthog)
    concat = classmethod(concat)

    def __init__(self, U, B, dtree, is_orthog=False):
        """
        Constructor: Create new htucker tensor with 'U' as basis matrices in the leaves and 'B' as transfer tensors
        in the inner nodes. Its tree structure is defined in 'dtree' and 'is_orthog' indicates if the
        htucker tensor is already orthogonal.
        @param U: dict. keys: non-negative integers, values: 2-D np.ndarrays
        @param B: dict. keys: non-negative integers, values: 3-D np.ndarrays
        @param dtree: tensor.utils.dimtree.dimtree
        @param is_orthog: bool
        """
        if not isinstance(U, dict):
            raise ValueError("'U' must be a dict with non-negative integers as keys and 2D np.ndarrays as values.")
        if not all(isinstance(v, np.ndarray) for v in U.values()):
            raise ValueError("'U' must be a dict with non-negative integers as keys and 2D np.ndarrays as values.")
        if not all(np.issubdtype(type(k), np.integer) for k in U.keys()):
            raise ValueError("'U' must be a dict with non-negative integers as keys and 2D np.ndarrays as values.")
        if not all(len(v.shape) == 2 for v in U.values()):
            raise ValueError("'U' must be a dict with non-negative integers as keys and 2D np.ndarrays as values.")
        if not isinstance(B, dict):
            raise ValueError("'B' must be a dict with non-negative integers as keys and 3D np.ndarrays as values.")
        if not all(isinstance(v, np.ndarray) for v in B.values()):
            raise ValueError("'B' must be a dict with non-negative integers as keys and 3D np.ndarrays as values.")
        if not all(np.issubdtype(type(k), np.integer) for k in B.keys()):
            raise ValueError("'B' must be a dict with non-negative integers as keys and 3D np.ndarrays as values.")
        if not all(len(v.shape) == 3 for v in B.values()):
            raise ValueError("'B' must be a dict with non-negative integers as keys and 3D np.ndarrays as values.")
        if not isinstance(dtree, dimtree):
            raise ValueError("'dtree' must be of type tensor.utils.dimtree.dimtree.")
        if not isinstance(is_orthog, bool):
            raise ValueError("If passed 'is_orthog' must be of type bool.")
        if not all(ind in U.keys() for ind in dtree.get_leaves()):
            raise ValueError("'U' and 'dtree' are not consistent.")
        if not all(ind in B.keys() for ind in dtree.get_inner_nodes()):
            raise ValueError("'B' and 'dtree' are not consistent.")

        # Set instance variables
        self.U = U
        self.B = B
        self.dtree = dtree
        self.shape = self.helper_get_shape()
        self.order = len(self.shape)
        self.rank = self.helper_get_rank()
        self.is_orthog = is_orthog

    def helper_get_shape(self):
        shape = []
        dt = self.dtree
        for node in dt.get_dim2ind():
            mode_size = self.U[node].shape[0]
            shape += [mode_size]
        return tuple(shape)

    def helper_get_rank(self):
        rank = {}
        for k, v in self.U.items():
            rank[k] = v.shape[1]
        for k, v in self.B.items():
            rank[k] = v.shape[2]
        return rank

    def update_shape(self):
        self.shape = self.helper_get_shape()

    def __getitem__(self, key):
        return self.get(key)
