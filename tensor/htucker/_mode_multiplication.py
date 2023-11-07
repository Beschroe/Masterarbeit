import numpy as np
from copy import deepcopy


def mode_multiplication(cls, x, A, mu):
    """
    Berechnet die Modusmultiplikation A o_mu x.
    @param x: htucker.HTucker
    @param A: 2D np.ndarray
    @param mu: nicht-negativer integer
    @return: htucker.HTucker
    """

    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not isinstance(A, np.ndarray):
        raise TypeError("'A' ist kein np.ndarray.")
    if not isinstance(mu, int):
        raise TypeError("'mu' ist kein integer.")
    if not len(A.shape) == 2:
        raise ValueError("'A' ist kein 2D-np.ndarray.")
    if not mu >= 0:
        raise ValueError("'mu' ist kein nicht-negativer integer.")
    if not mu < len(x.shape):
        raise ValueError("'mu' passt nicht zu 'x', da es zu groß ist.")
    if not x.shape[mu] == A.shape[1]:
        raise ValueError("'x', 'A' und 'mu' passen nicht zusammen.")

    dt = x.dtree.copy()
    U = deepcopy(x.U)
    B = deepcopy(x.B)

    # Index des Modus 'mu'
    ind = dt.get_ind(mu)

    # Modusmultiplizieren von 'A' und 'x'
    U[ind] = np.dot(A, U[ind])

    # Resultierender hierarchischer Tuckertensor
    z = cls(U=U, B=B, dtree=dt, is_orthog=False)
    return z
