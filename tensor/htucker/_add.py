import numpy as np
from tensor.utils.dimtree import equal


def add(cls, x, y):
    """
    Addiert die beiden hierarchischen Tuckertensoren 'x' und 'y', wobei identische Dimensionsbäume vorausgesetzt werden.
    @param x: htucker.HTucker
    @param y: htucker.HTucker
    @return: htucker.HTucker
    """
    # Argument checks
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not isinstance(y, cls):
        raise TypeError("'y' ist kein hierarchischer Tuckertensor.")
    if not x.shape == y.shape:
        raise ValueError("Die Modi von 'x' und 'y' stimmen nicht überein.")
    if not equal(x.dtree, y.dtree):
        raise ValueError("Die Dimensionsbäume von 'x' und 'y' sind nicht identisch.")

    # 1) Neuer Dimensionsbaum
    dtree = x.dtree.copy()

    # 2) Konkateniere die Blattmatrizen von 'x' und 'y'
    U = {}
    for leaf in dtree.get_leaves():
        U[leaf] = np.hstack((x.U[leaf], y.U[leaf]))

    # 3) Ränge von x und y
    kx = x.rank
    ky = y.rank

    # 4) Konstruiere die neuen Transfertensoren
    B = {}
    for t in dtree.get_inner_nodes():
        if t == 0:
            # Ueberspringe Wurzel Fall
            continue
        tl = dtree.get_left(t)
        tr = dtree.get_right(t)
        # Rang: k_tl x k_tr x k_t
        k_t = kx[t] + ky[t]
        k_tl = kx[tl] + ky[tl]
        k_tr = kx[tr] + ky[tr]
        # Konstruiere Transfertensor
        Bt = np.zeros((k_tl, k_tr, k_t))
        Bt[:kx[tl], :kx[tr], :kx[t]] = x.B[t]
        Bt[kx[tl]:, kx[tr]:, kx[t]:] = y.B[t]
        B[t] = Bt

    # Wurzelfall
    tl = dtree.get_left(0)
    tr = dtree.get_right(0)
    # Rang: k_tl x k_tr x1
    k_t = 1
    k_tl = kx[tl] + ky[tl]
    k_tr = kx[tr] + ky[tr]
    # Konstruiere Transfertensor
    Bt = np.zeros((k_tl, k_tr, k_t))
    Bt[:kx[tl], :kx[tr]] = x.B[0]
    Bt[kx[tl]:, kx[tr]:] = y.B[0]
    B[0] = Bt

    # Erstelle darauf aufbauend resultierenden HTucker Tensor
    z = cls(U=U, B=B, dtree=dtree, is_orthog=False)
    return z
