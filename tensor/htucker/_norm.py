import numpy as np


def norm(cls, x):
    """
    Berechnet die Norm des hierarchischen Tuckertensors 'x'.
    @param x: htucker.HTucker
    @return: float
    """
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")

    if not x.is_orthog:
        x = cls.orthogonalize(x)
    B = x.B
    B_root = B[0]
    frob_norm = np.linalg.norm(B_root)
    return frob_norm
