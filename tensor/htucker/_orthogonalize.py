import numpy as np
from copy import deepcopy


def orthogonalize(cls, x):
    """
    Orthogonalisiert eine Kopie von 'x'.
    @param x: htucker.HTucker
    @return: htucker.HTucker
    """
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")

    # Falls 'x' bereits orthogonal ist, gebe eine Kopie zurück
    if x.is_orthog:
        return deepcopy(x)

    # Initialisiere Dimensionsbaum, Blattmatrizen und Transfertensoren des neuen hierarchischen Tuckertensors
    dt = x.dtree.copy()
    U_upd = {}
    B_upd = {}
    R = {}

    # Anpassen der Blattmatrizen
    for leaf in dt.get_leaves():
        Ut = x.U[leaf]
        Ut_orthog, Rt = np.linalg.qr(Ut)
        U_upd[leaf] = Ut_orthog
        R[leaf] = Rt

    # Anpassen der Transfertensoren
    depth = dt.get_depth()
    for level in range(depth - 1, -1, -1):
        for node in dt.get_nodes_of_level(level):
            if dt.is_leaf(node):
                continue
            right = dt.get_right(node)
            left = dt.get_left(node)
            Rtr = R[right]
            Rtl = R[left]
            Bt_hat = np.tensordot(Rtr, x.B[node], axes=[1, 1])
            Bt_hat = np.tensordot(Rtl, Bt_hat, axes=[1, 1])
            if node == 0:
                B_upd[node] = Bt_hat
            else:
                Bt_hat = Bt_hat.reshape((Bt_hat.shape[0] * Bt_hat.shape[1], -1), order="F")
                Bt_upd, Rt = np.linalg.qr(Bt_hat)
                Bt_upd = Bt_upd.reshape((Rtl.shape[0], Rtr.shape[0], -1), order="F")
                B_upd[node] = Bt_upd
                R[node] = Rt
            del R[right]
            del R[left]
    orthog_htucker = cls(U=U_upd, B=B_upd, dtree=dt, is_orthog=True)
    return orthog_htucker
