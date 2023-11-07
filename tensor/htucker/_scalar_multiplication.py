import numpy as np
from copy import deepcopy


def scalar_mul(cls,x, a):
    """
    Berechnet die Skalarmultiplikation des floats/ints 'a' mit dem hierarchischen Tuckertensor
    'x'.
    :param x: htucker.HTucker
    :param a: float oder int
    :return: htucker.HTucker
    """
    # Argument Checks
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not np.issubdtype(type(a), np.integer) and not np.issubdtype(type(a),np.float):
        raise TypeError("'a' ist weder int noch float.")

    z = deepcopy(x)
    z.B[0] = a * z.B[0]
    return z
