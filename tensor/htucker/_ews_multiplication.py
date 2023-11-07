from tensor.arithmetics.multilinear_mul import multi_mul
from tensor.arithmetics.left_svd_gramian import left_svd_gramian
from tensor.utils.dimtree import equal
from copy import deepcopy
import numpy as np


def ews_multiplication(cls, x, y, max_rank, abs_err=None):
    """
    Elementweise Multiplikation zweier hierarchischer Tuckertensoren 'x' und 'y'.
    Dies ist nur möglich, falls 'x' und 'y' identische Dimensionsbäume aufweisen.
    Während der Berechnung wird parallel eine Kürzung entsprechend 'max_rank' und 'abs_err' durchgeführt.
    @param x: tensor.htucker.htucker
    @param y: tensor.htucker.htucker
    @param max_rank: positive integer
    @param abs_err: positive float
    @return: tensor.htucker.htucker, dict, dict
    """
    # Check arguments
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not isinstance(y, cls):
        raise TypeError("'y' ist kein hierarchischer Tuckertensor.")
    if not x.shape == y.shape:
        raise ValueError("'x' und 'y' haben nicht die gleichen Modi.")
    if not equal(x.dtree, y.dtree):
        raise ValueError("Die Dimensionsbäume von 'x' und 'y' sind nicht identisch.")
    if not np.issubdtype(type(max_rank), int):
        raise TypeError("'max_rank' ist kein integer.")
    if not max_rank > 0:
        raise ValueError("'max_rank' ist kein positiver integer.")
    if abs_err is not None:
        if not np.issubdtype(type(abs_err), float):
            raise TypeError("'abs_err' ist kein float.")
        if not abs_err > 0:
            raise ValueError("'abs_err' ist kein positiver float.")

    # Orthogonalisiere 'x' und 'y'
    x = cls.orthogonalize(x)
    y = cls.orthogonalize(y)

    # Berechne reduzierten Gram'sche Matrizen
    Gx = cls.gramians_orthog(x)
    Gy = cls.gramians_orthog(y)

    # Initialisiere Variablen des resultierenden HTucker Tensors
    dtree_z = deepcopy(x.dtree)
    Uz = {}
    Bz = {}

    # Berechne knotenweise Fehlerschranke
    if abs_err is not None:
        abs_err = abs_err / np.sqrt(2 * x.order - 2)

    # Knotenweise Fehler und Singulärwerte
    err = {}
    sv = {}

    # Gekürzte Blattmatrizen
    U_x = {}
    U_y = {}

    for t in range(1, dtree_z.get_nr_nodes()):
        # Berechne linke Singulärvektoren und Singulärwerte
        ux, sx = left_svd_gramian(Gx[t])
        uy, sy = left_svd_gramian(Gy[t])

        # Berechne alle Kombinationen an Singulärwerten
        sz = sx.reshape((-1,1)) @ sy.reshape((-1,1)).T
        idcz = np.argsort(sz.ravel(order="F"))[::-1]
        sv[t] = sz.ravel(order="F")[idcz]

        # Bestimme erforderlichen Rang
        k, err[t], _ = cls.trunc_rank(sv[t], max_rank=max_rank, abs_err=abs_err, rel_err=None)

        # Berechne die Indizes der entsprechenden linken Singulärvektoren
        ind_x, ind_y = np.unravel_index(indices=idcz[:k], shape=sz.shape, order="F")

        # Behalte die ausgewählten linken Singulärvektoren
        U_x[t] = ux[:, ind_x]
        U_y[t] = uy[:, ind_y]

    for t in range(dtree_z.get_nr_nodes()-1, 0, -1):
        if dtree_z.is_leaf(t):
            # neue Blattmatrix ist elementweises Produkt der gekürzten Blattmatrizen
            Ux = x.U[t] @ U_x[t]
            Uy = y.U[t] @ U_y[t]
            Uz[t] = Ux * Uy
        else:
            # Neuer Transfertensor ist elementweises Produkt der gekürzten Transfertensoren
            ii_left = dtree_z.get_left(t)
            ii_right = dtree_z.get_right(t)
            Bx = multi_mul(x.B[t], [U_x[ii_left].T, U_x[ii_right].T, U_x[t].T], [0,1,2])
            By = multi_mul(y.B[t], [U_y[ii_left].T, U_y[ii_right].T, U_y[t].T], [0, 1, 2])
            Bz[t] = Bx * By

    # Wurzelfall
    ii_left = dtree_z.get_left(0)
    ii_right = dtree_z.get_right(0)
    Bx = multi_mul(x.B[0], [U_x[ii_left].T, U_x[ii_right].T], [0, 1])
    By = multi_mul(y.B[0], [U_y[ii_left].T, U_y[ii_right].T], [0, 1])
    Bz[0] = Bx * By

    # Erzeuge resultierenden HTucker Tensor
    z = cls(U=Uz, B=Bz, dtree=dtree_z, is_orthog=False)
    return z, err, sv



