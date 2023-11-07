import numpy as np
from tensor.arithmetics.mode_multiplication import mode_multiplication


def gramians_orthog(cls, x):
    """
    Berechnet die Gram'schen Matrizen für einen orthogonalen hierarchischen Tuckertensor.
    @param x: tensor.htucker.htucker
    @return: dict bestehend aus 2-D np.ndarrays
    """
    # Argument check
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")

    # Prüfe ob x orthogonal ist
    if not x.is_orthog:
        raise ValueError("'x' ist nicht orthogonal.")

    # Memorize gramians in dict
    # The roots gramian is 1
    G = {0: np.ones((1, 1))}

    # Traverse tree top down
    for level in range(0, x.dtree.get_depth(), 1):
        for t in x.dtree.get_nodes_of_level(level):
            if x.dtree.is_leaf(t):
                continue
            # Children
            t_left = x.dtree.get_left(t)
            t_right = x.dtree.get_right(t)

            B_mod = mode_multiplication(U=G[t], A=x.B[t], mu=2)

            G[t_left] = np.tensordot(x.B[t], B_mod, axes=[[1, 2], [1, 2]])
            G[t_right] = np.tensordot(x.B[t], B_mod, axes=[[0, 2], [0, 2]])

    return G

