from tensor.arithmetics.left_svd_gramian import left_svd_gramian
import numpy as np

def truncate_htucker(cls,x, max_rank, abs_err=None, rel_err=None):
    """
    Kürzt einen gegebenen hierarchischen Tuckertensor 'x' auf einen gegebenen kleinerer hierarchischen
    Rang entsprechend 'max_rank', 'abs_err' und 'rel_err'.
    Hierbei dominiert 'max_rank' die beiden Fehlertoleranzen.
    """
    # Argument Checks
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not np.issubdtype(type(max_rank), int):
        raise TypeError("'max_rank' muss ein positiver int sein.")
    if max_rank < 1:
        raise ValueError("'max_rank' muss ein positiver int sein.")
    if abs_err is not None:
        if not np.issubdtype(type(abs_err), float):
            raise TypeError("'abs_err' muss ein positiver float sein.")
        if not abs_err > 0:
            raise ValueError("'abs_err' muss ein positiver float sein.")
    if rel_err is not None:
        if not np.issubdtype(type(rel_err), float):
            raise TypeError("'rel_err' muss ein positiver float sein.")
        if not rel_err > 0:
            raise ValueError("'rel_err' muss ein positiver float sein.")

    # Orthogonalisiere x
    x = cls.orthogonalize(x)

    # Berechne reduzierte Gram'sche Matrizen
    G = cls.gramians_orthog(x)

    # Knotenweise Fehler und Singulärwerte
    err = {}
    sv = {}

    # Berechne knotenweise Fehlerschranke
    if abs_err is not None:
        abs_err = abs_err / np.sqrt(2 * x.order - 2)
    if rel_err is not None:
        rel_err = rel_err / np.sqrt(2 * x.order - 2)

    # Hierarchischer Rang
    rank = {}

    # Berechne die linken Singulärvektoren der reduzierten Gram'schen Matrizen
    U = {}
    for ii in range(1, x.dtree.get_nr_nodes()):
        u, sv[ii] = left_svd_gramian(G[ii])
        sv[ii] = sv[ii].reshape((-1))
        # Berechne Rang in Abhängigkeit von max_rank und abs_err sowie rel_err
        k, err[ii], sat = cls.trunc_rank(sv[ii], max_rank, abs_err=abs_err, rel_err=rel_err)
        rank[ii] = k
        # Behalte nur die k dominaten Singulärvektoren
        U[ii] = u[:, :k]

    # Berechne die gekürzten Blattmatrizen
    U_new = {}
    for ii in x.dtree.get_leaves():
        U_new[ii] = x.U[ii] @ U[ii]

    U[0] = np.ones((1, 1))
    rank[0] = 1
    # Berechne die gekürzten Transfertensoren
    B_new = {}
    for ii in x.dtree.get_inner_nodes():
        ii_left = x.dtree.get_left(ii)
        ii_right = x.dtree.get_right(ii)
        U_l = U[ii_left]
        U_r = U[ii_right]
        # Berechne (U_r.T kron U_l.T) @ x.B[ii] @ U[ii]
        product = np.tensordot(x.B[ii], U[ii], axes=[2, 0])
        product = np.tensordot(U_r.T, product, axes=[1, 1])
        product = np.tensordot(U_l.T, product, axes=[1, 1])
        B_new[ii] = product

    # Erzeuge den resultierenden gekürzten HTucker Tensor
    dtree = x.dtree.copy()
    new_htucker = cls(U=U_new, B=B_new, dtree=dtree, is_orthog=False)

    return new_htucker, err, sv




