import numpy as np
from tensor.utils.dimtree import equal


def inner_product(cls, x, y):
    """
    Berechnet das innere Produkt zweier hierarchischer Tuckertensoren 'x' und 'y' mit übereinstimmenden Dimensionsbäumen.
    @param x: tensor.htucker.htucker
    @param y: tensor.htucker.htucker
    @return: float
    """

    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not isinstance(y, cls):
        raise TypeError("'y' ist kein hierarchischer Tuckertensor.")
    if not equal(x.dtree, y.dtree):
        raise ValueError("Die Dimensionsbäume von 'x' und 'y' sind nicht identisch.")

    dt = x.dtree
    M = {}

    for t in dt.get_leaves():
        M[t] = x.U[t].T @ y.U[t]
    depth = dt.get_depth()
    for level in range(depth - 1, -1, -1):
        for t in dt.get_nodes_of_level(level):
            if dt.is_leaf(t):
                # t ist ein Blatt
                continue
            right = dt.get_right(t)
            left = dt.get_left(t)
            # Berechne Mt
            Mt = np.tensordot(M[left], y.B[t], axes=[1, 0])
            Mt = np.tensordot(M[right], Mt, axes=[1, 1])
            Mt = np.tensordot(x.B[t], Mt, axes=[[0, 1], [1, 0]])
            M[t] = Mt
    return M[0]
