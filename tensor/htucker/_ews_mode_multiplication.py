import numpy as np
from copy import deepcopy


def ews_mode_multiplication(cls, x, vec, mu):
    """
    Elementweise Modusmultiplikation vec *_mu x
    @param x: htucker.HTucker
    @param vec: 1-D np.ndarray
    @param mu: nicht-negativer integer
    @return: htucker.HTucker
    """

    # Argument Checks
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not isinstance(vec, np.ndarray):
        raise TypeError("'vec' ist kein np.ndarray.")
    if len(vec.shape) != 1:
        raise ValueError("'vec' ist kein 1D-np.ndarray.")
    if not np.issubdtype(type(mu), int):
        raise TypeError("'mu' ist kein integer.")
    if mu not in range(x.order):
        raise ValueError("'mu' ist kein integer aus {0, ...,d-1} wobei d die Ordnung von 'x' ist.")
    if x.shape[mu] != vec.shape[0]:
        raise ValueError("'x', 'vec' und 'mode' passen nicht zusammen.")

    dt = x.dtree.copy()
    U = deepcopy(x.U)
    B = deepcopy(x.B)

    # Multipliziere vec elementweise zu der Blattmatrix U[mu]
    leaf_index = dt.get_ind(mu)
    U[leaf_index] = U[leaf_index] * vec[:, None]

    # Konstruiere resultierenden neuen hierarchischen Tuckertensor
    z = cls(U=U, B=B, dtree=dt, is_orthog=False)
    return z
