import numpy as np
from tensor.utils.dimtree import equal
from tensor.arithmetics.multilinear_mul import multi_mul
from tensor.arithmetics.left_svd_gramian import left_svd_gramian
from copy import deepcopy
from tensor.transformation.matricise import matricise


def add_and_truncate(cls, summanden, max_rank, abs_err=None, rel_err=None, copy=True):
    """
    Addiert die hierarchischen Tuckertensoren aus 'summanden' und kürzt gleichzeitig
    den hierarchischen Rang gemäß 'max_rank' und den übergebenen Fehlertoleranzen.
    Hinweis: Die zu addierenden hierarchischen Tuckertensoren müssen über identische Dimensionsbäume verfügen.
    @param summanden: list aus htucker.HTucker Objekten
    @param max_rank: positiver int
    @param abs_err: positiver float
    @param rel_err: positiver float
    @param copy: bool
    @return: htucker.HTucker
    """

    # Argument checks
    if not isinstance(summanden, list):
        raise TypeError("'summanden' ist keine Liste.")
    if not all(isinstance(element, cls) for element in summanden):
        raise TypeError("Nicht alle Elemente aus 'summanden' sind hierarchische Tuckertensoren.")
    if not all(element.shape == summanden[0].shape for element in summanden):
        raise ValueError("Nicht alle hierarchische Tuckertensoren aus 'summanden' haben dieselben Modi.")
    if not all(equal(element.dtree, summanden[0].dtree) for element in summanden):
        raise ValueError("Nicht alle hierarchische Tuckertensoren aus 'summanden' haben identische Dimensionsbäume.")
    if not np.issubdtype(type(max_rank), int):
        raise TypeError("'max_rank' muss ein int>= 1 sein.")
    if not max_rank >= 1:
        raise ValueError("'max_rank' muss ein int>= 1 sein.")
    if abs_err is not None:
        if not np.issubdtype(type(abs_err), float):
            raise TypeError("'abs_err' muss ein float > 0 sein.")
        if not abs_err > 0:
            raise ValueError("'abs_err' muss ein float > 0 sein.")
    if rel_err is not None:
        if not np.issubdtype(type(rel_err), float):
            raise TypeError("'rel_err' muss ein float > 0 sein.")
        if not rel_err > 0:
            raise ValueError("'rel_err' muss ein float > 0 sein.")
    if not isinstance(copy, bool):
        raise TypeError("'copy' muss ein bool sein.")

    if copy:
        summanden = deepcopy(summanden)

    # Variablen des resultierenden Htucker Tensors
    dtreez = summanden[0].dtree
    Uz = {}
    Bz = {}

    # Dicts für die knotenweisen QR-Zerlegungen
    Q = {}
    R = {}

    # Berechne knotenweise Fehlerschranke
    if abs_err is not None:
        abs_err = abs_err / np.sqrt(2 * summanden[0].order - 2)
    if rel_err is not None:
        rel_err = rel_err / np.sqrt(2 * summanden[0].order - 2)

    # Berechne QR-Zerlegung für jede Blattmatrix
    for t in dtreez.get_leaves():
        # Konkatenieren der t-Blattmatrizen aller Summanden
        Uz[t] = np.hstack([htensor.U[t] for htensor in summanden])
        # QR Zerlegung
        Q[t], R[t] = np.linalg.qr(Uz[t], mode="reduced")
        Uz[t] = R[t]

    # Berechnung der reduzierten Gram'schen Matrizen der (impliziten) Summe
    G = cls.gramians_sum(summanden)

    # Knotenweise Fehler und Singulärwerte
    err = {}
    sv = {}

    # Durchschreiten des Dimensionsbaums von unten nach oben
    for level in range(dtreez.get_depth(), 0, -1):
        for t in dtreez.get_nodes_of_level(level):

            if dtreez.is_leaf(t):
                Uz[t] = Q[t]

            else:
                # Kinder
                l, r = dtreez.get_left(t), dtreez.get_right(t)
                # Teilt R[l] und. R[r] auf in [R[l]_1, R[l]_2,...,R[l]_d]
                # bzw. [R[r]_1, R[r]_2,...,R[r]_d], wobei d die Anzahl
                # an Summanden ist
                R_left = np.split(R[l], np.cumsum([htensor.rank[l] for htensor in summanden[:-1]]), axis=1)
                R_right = np.split(R[r], np.cumsum([htensor.rank[r] for htensor in summanden[:-1]]), axis=1)

                Bz[t] = []
                for jj in range(len(summanden)):
                    Bzt = np.tensordot(R_right[jj], summanden[jj].B[t], axes=[1, 1])
                    Bz[t] += [np.tensordot(R_left[jj], Bzt, axes=[1, 1])]

                Bz[t] = np.concatenate(Bz[t], axis=2)

                # Matriziere Bz[t]
                B_mat = matricise(Bz[t], t=[0, 1])

                # Berechne QR-Zerlegung
                q, r = np.linalg.qr(B_mat, mode="reduced")
                R[t] = r

                # Berechne shape des 'tensors' q
                shape_new = np.array(Bz[t].shape)
                shape_new[2] = q.shape[1]

                # Dematriziere Q zu Bz[t]
                Bz[t] = q.reshape(shape_new, order="F")

            # Update reduzierte Grammatrix
            G[t] = R[t] @ G[t] @ R[t].T

            # Berechne linke Singulärvektoren samt Singulärwerten
            u, sv[t] = left_svd_gramian(G[t])

            # Berechne den Rang, auf den gekürzt werden soll
            k, err[t], sat = cls.trunc_rank(s=sv[t], max_rank=max_rank,
                                            abs_err=abs_err, rel_err=rel_err)

            # Nehme entsprechend nur die k ersten Spalten mit
            u = u[:, :k]
            if dtreez.is_leaf(t):
                Uz[t] = Uz[t] @ u
            else:
                Bz[t] = np.tensordot(Bz[t], u.T, axes=[2, 1])

            R[t] = u.T @ R[t]

    # Anwendung von R auf die Wurzel
    # Kinder
    l = dtreez.get_left(0)
    r = dtreez.get_right(0)
    # Teilt R[l] und. R[r] auf in [R[l]_1, R[l]_2,...,R[l]_d]
    # bzw. [R[r]_1, R[r]_2,...,R[r]_d], wobei d die Anzahl
    # an Summanden ist
    R_left = np.split(R[l], np.cumsum([htensor.rank[l] for htensor in summanden[:-1]]), axis=1)
    R_right = np.split(R[r], np.cumsum([htensor.rank[r] for htensor in summanden[:-1]]), axis=1)
    Bz[0] = multi_mul(x=summanden[0].B[0], U=[R_left[0], R_right[0]], modes=[0, 1])
    for jj in range(1, len(summanden)):
        Bz[0] += multi_mul(x=summanden[jj].B[0], U=[R_left[jj], R_right[jj]], modes=[0, 1])

    z = cls(U=Uz, B=Bz, dtree=dtreez, is_orthog=True)
    return z, err, sv
