import numpy as np

from tensor.utils.dimtree import dimtree


class HTucker:
    """
    Implementierung des hierarchischen Tuckerformats.
    """
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

    def __init__(self, U, B, dtree, is_orthog=False):
        """
        Konstruktur: Die Blattmatrizen der neuen HTucker Instanz sind durch das dict 'U' gegeben. Die Transfertensoren
        sind im dict 'B' enthalten. 'dtree' entspricht dem kanonischen Dimensionsbaum und 'is_orthog' zeigt an, ob es
        sich um einen orthogonalen HTucker Tensor handelt.
        @param U: dict. keys: nicht-negative ints, values: 2D-np.ndarrays
        @param B: dict. keys: nicht-negative ints, values: 3D-np.ndarrays
        @param dtree: tensor.utils.dimtree.dimtree
        @param is_orthog: bool
        """
        if not isinstance(U, dict):
            raise TypeError("'U' muss ein dict mit nicht-negativen ints als keys und 2D-np.ndarrays als values sein.")
        if not all(isinstance(v, np.ndarray) for v in U.values()):
            raise ValueError("'U' muss ein dict mit nicht-negativen ints als keys und 2D-np.ndarrays als values sein.")
        if not all(np.issubdtype(type(k), np.integer) for k in U.keys()):
            raise ValueError("'U' muss ein dict mit nicht-negativen ints als keys und 2D-np.ndarrays als values sein.")
        if not all(len(v.shape) == 2 for v in U.values()):
            raise ValueError("'U' muss ein dict mit nicht-negativen ints als keys und 2D-np.ndarrays als values sein.")
        if not isinstance(B, dict):
            raise TypeError("'B' muss ein dict mit nicht-negativen ints als keys und 3D-np.ndarrays als values sein.")
        if not all(isinstance(v, np.ndarray) for v in B.values()):
            raise ValueError("'B' muss ein dict mit nicht-negativen ints als keys und 3D-np.ndarrays als values sein.")
        if not all(np.issubdtype(type(k), np.integer) for k in B.keys()):
            raise ValueError("'B' muss ein dict mit nicht-negativen ints als keys und 3D-np.ndarrays als values sein.")
        if not all(len(v.shape) == 3 for v in B.values()):
            raise ValueError("'B' muss ein dict mit nicht-negativen ints als keys und 3D-np.ndarrays als values sein.")
        if not isinstance(dtree, dimtree):
            raise TypeError("'dtree' muss vom Typ tensor.utils.dimtree.dimtree sein.")
        if not isinstance(is_orthog, bool):
            raise TypeError("Falls übergeben, muss 'if_orthog vom Typ bool sein.")
        if not all(ind in U.keys() for ind in dtree.get_leaves()):
            raise ValueError("'U' und 'dtree' sind nicht konsistent.")
        if not all(ind in B.keys() for ind in dtree.get_inner_nodes()):
            raise ValueError("'B' und 'dtree' sind nicht konsistent.")

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
