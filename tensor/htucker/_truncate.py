import numpy as np
from tensor.transformation.matricise import matricise
from tensor.arithmetics.multilinear_mul import multi_mul
from tensor.utils.dimtree import dimtree

def truncate(cls, A, max_rank, abs_err=None, rel_err=None, dtree=None):
    """
    Berechnet das hierarchische Tuckerformat für den vollen Tensor 'A' unter Einhaltung des in 'max_rank' festgelegten
    maximalen hierarchischen Ranges und den in 'abs_err' und 'rel_err' definierten Fehlerschranken. Im Zweifel dominiert
    'max_rank' jedoch die Fehlerschranken.
    Zurückgegeben werden der resultierende hierarchische Tuckertensor, die knotenweisen Fehler und die knotenweisen
    Singulärwerte.
    @param A: N-D np.ndarray with N >= 1
    @param dtree: tensor.utils.dimtree.dimtree
    @param max_rank: positive integer
    @param abs_err: positive float
    @param rel_err: positive float
    @return: tensor.htucker.htucker, dict, dict
    """

    # Check arguments
    if not isinstance(A, np.ndarray):
        raise TypeError("'A' muss ein ND-np.ndarray mit N >= 1 sein.")
    if not len(A.shape) >= 1:
        raise ValueError("'A' muss ein ND-np.ndarray mit N >= 1 sein.")
    if not np.issubdtype(type(max_rank), int):
        raise TypeError("'max_rank' muss ein positiver int sein.")
    if not max_rank >= 1:
        raise ValueError("'max_rank' muss ein positiver int sein.")
    if abs_err is not None:
        if not np.issubdtype(type(abs_err), float):
            raise TypeError("'abs_err' muss ein nicht-negativer float sein.")
        if not abs_err > 0:
            raise ValueError("'rel_err' muss ein nicht-negativer float sein.")
    if rel_err is not None:
        if not np.issubdtype(type(rel_err), float):
            raise TypeError("'abs_err' muss ein nicht-negativer float sein.")
        if not rel_err > 0:
            raise ValueError("'rel_err' muss ein nicht-negativer float sein.")
    if dtree is not None:
        raise ValueError("Falls übergeben muss 'dtree' vom Typ tensor.utils.dimtree.dimtree sein.")

    # Initialisiere Dimensionsbaum
    if dtree is None:
        # Erzeuge neuen Dimensionsbaum
        dtree = dimtree.get_canonic_dimtree(len(A.shape))
    else:
        # Prüfe den übergebenen Dimensionsbaum auf Konsistenz mit 'A'
        if dtree.get_nr_dims() != len(A.shape):
            raise ValueError("'A' und 'dtree' sind nicht konsistent.")

    # Berechne die knotenweise Fehlertoleranz
    if abs_err is not None:
        abs_err = abs_err / np.sqrt(2 * dtree.get_nr_dims() - 2)
    if rel_err is not None:
        rel_err = rel_err / np.sqrt(2 * dtree.get_nr_dims() - 2)

    # Initialisiere die Variablen des hierarchischen Tuckertensors
    U_leaves = {}    # Blattmatrizen
    B = {}           # Transfertensoren
    is_orthog = True
    p = dtree.get_depth()    # Tiefe des Dimensionsbaums
    node_to_dim = {}         # Mapping von Knotenindex zu repräsentierten Dimensionen
    rank = {}                # Hierarchischer Rang
    error = {}               # Knotenweise eingehaltene Fehlerschranken
    sv = {}                  # Knotenweise Singulärwerte der entsprechenden Matrizierungen

    for leaf in dtree.get_leaves():
        # Blattmatrizen
        dim = dtree.get_dim(leaf)
        A_leaf = matricise(A, list(dim))
        # Singulärwertzerkegubg
        U_leaves[leaf], sv[leaf], _ = np.linalg.svd(A_leaf, full_matrices=False)
        # Bestimme notwendigen Rang, um die Fehlertoleranzen einzuhalten
        rank[leaf], error[leaf], sat = cls.trunc_rank(sv[leaf], max_rank=max_rank, abs_err=abs_err, rel_err=rel_err)
        # Behalte nur die ersten k Spalten mit
        U_leaves[leaf] = U_leaves[leaf][:, :rank[leaf]]
        # Aktualisieren des Knoten zu Dimension Mappings
        dims = dtree.get_dim(leaf)
        node_to_dim[leaf] = dims[0]

    # Kürze 'A' auf den Kerntensor C
    # Alle weiteren Berechnungen basieren auf C
    U = [U_leaves[k].T for k in sorted(list(U_leaves.keys()), key=lambda x: dtree.get_dim(x)[0])]
    modes = list(range(len(A.shape)))
    C = multi_mul(x=A, U=U, modes=modes)

    # Traversiere den Dimensionsbaum von unten nach oben
    for level in range(p - 1, 0, -1):

        Cl = np.array(C)
        node_to_dim_new = dict(node_to_dim)
        for t in dtree.get_nodes_of_level(level):
            # Berechne den Transfertensor des Knotens t
            if dtree.is_leaf(t):
                # t ist ein Blatt und die Blattmatrizen wurden bereits berechnet
                continue
            # Matriziere C mit den Modi aus t als Zeilen
            # Die Modi, die t repräsentiert, sind wiederum durch ts Kinder gegeben
            left = node_to_dim[dtree.get_left(t)]
            right = node_to_dim[dtree.get_right(t)]
            C_matricised = matricise(C, [left, right])
            # Singulärwertzerlegung
            u_t, sv[t], _ = np.linalg.svd(C_matricised, full_matrices=False)
            # Berechne notwendigen Rang k, um die Fehlertoleranzen einzuhalten
            rank[t], error[t], sat = cls.trunc_rank(sv[t], max_rank=max_rank, abs_err=abs_err, rel_err=rel_err)
            # Behalte die ersten k Spalten
            u_t = u_t[:, :rank[t]]
            # Der auf 3D umgeformte Tensor basierend auf u_t entspricht nun dem Transfertensor
            B[t] = u_t.reshape((rank[dtree.get_left(t)], rank[dtree.get_right(t)], rank[t]), order="F")
            # Darauf aufbauend wird nun der Transfertensor Cl weiter gekürzt
            left_dim_already_updated = node_to_dim_new[dtree.get_left(t)]
            right_dim_already_updated = node_to_dim_new[dtree.get_right(t)]
            Cl = np.tensordot(B[t], Cl, axes=[[0, 1], [left_dim_already_updated, right_dim_already_updated]])
            Cl = np.moveaxis(Cl, source=0, destination=left_dim_already_updated)
            # Aktualisiere das Knotenindex zu Dimension Mapping
            mini = min(left_dim_already_updated, right_dim_already_updated)
            maxi = max(left_dim_already_updated, right_dim_already_updated)
            dict_buf = {k: (v - 1 if v > maxi else v) for k, v in node_to_dim_new.items()}
            del dict_buf[dtree.get_left(t)]
            del dict_buf[dtree.get_right(t)]
            dict_buf[t] = mini
            node_to_dim_new = dict_buf
        # Der Kerntensor C wird auf den in obiger Schleife gekürzten Kerntensor Cl gesetzt
        # Das alte Knotenindex zu Dimensionen Mapping wird durch das aktualisierte ersetzt
        node_to_dim = node_to_dim_new
        C = Cl
    # Im letzten Schritt wird nun die Wurzel behandelt
    # Dazu wird C in Bezug auf die Modi der Wurzel matriziert, was ein einer Vektorisierung entspricht, da die
    # Wurzel alle Modi repräsentiert
    # Erneut ist es dazu notwendig, die Indizes des Kinder der Wurzel in Bezug auf C zu kennen
    left = node_to_dim[dtree.get_left(0)]
    right = node_to_dim[dtree.get_right(0)]
    # Matrizierung/Vektorisierung
    C_root = matricise(C, [left, right])
    rank[0] = 1
    # Der Transfertensor entspricht in diesem Fall einer Matrix
    k_l = rank[dtree.get_left(0)]
    k_r = rank[dtree.get_right(0)]
    B[0] = C_root.reshape((k_l, k_r, 1), order="F")

    # Erzeuge darauf aufbauend den resultierenden hierarchischen Tuckertensor
    Ah = cls(U=U_leaves, B=B, dtree=dtree, is_orthog=is_orthog)
    return Ah, error, sv
