import numpy as np
from tensor.arithmetics.mode_multiplication import mode_multiplication
from tensor.utils.dimtree import dimtree
from copy import deepcopy


def change_dimtree(cls, x, children, dim2ind):
    """
    Gibt eine Kopie von 'x' zurück, deren Dimensionsbaum gemäß 'children' und 'dim2ind' modifiziert wurde.
    @param x: htucker.HTucker
    @param children: list oder np.ndarray
    @param dim2ind: list, tuple oder np.ndarray
    @return: htucker.HTucker
    """

    # Argument Checks
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischen Tuckertensor.")
    if not isinstance(children, list) and not isinstance(children, np.ndarray):
        raise TypeError("children muss eine list oder np.ndarray sein. \n Es muss von folgender Form sein:"
                        " [[a,b], [c,d], ... [x,y]],\n wobei a,b,c,... ints >= -1 sind")
    if isinstance(children, list):
        children = np.array(children)
    if not all(len(kids) == 2 for kids in children):
        raise ValueError("children muss eine list oder np.ndarray sein. \n Es muss von folgender Form sein:"
                         " [[a,b], [c,d], ... [x,y]],\n wobei a,b,c,... ints >= -1 sind")
    if not all(kids[0] >= -1 and kids[1] >= -1 for kids in children):
        raise ValueError("children muss eine list oder np.ndarray sein. \n Es muss von folgender Form sein:"
                         " [[a,b], [c,d], ... [x,y]],\n wobei a,b,c,... ints >= -1 sind")
    if not isinstance(dim2ind, list) and not isinstance(dim2ind, np.ndarray) \
            and not isinstance(dim2ind, tuple):
        raise TypeError("dim2ind must be of type list, np.ndarray or tuple and contain \n"
                        "integers n with 0 <= n < len(children). Furthermore is must not \n"
                        "contain any duplicates.")
    if isinstance(dim2ind, list) or isinstance(dim2ind, tuple):
        dim2ind = np.array(dim2ind)
    if not all(ind < len(children) for ind in dim2ind):
        raise ValueError("dim2ind must be of type list, np.ndarray or tuple and contain \n"
                         "integers n with 0 <= n < len(children). Furthermore is must not \n"
                         "contain any duplicates.")
    if not len(dim2ind.shape) == 1:
        raise ValueError("dim2ind must be of type list, np.ndarray or tuple and contain \n"
                         "integers n with 0 <= n < len(children). Furthermore is must not \n"
                         "contain any duplicates.")
    if len(set(dim2ind)) < len(dim2ind):
        raise ValueError("dim2ind must be of type list, np.ndarray or tuple and contain \n"
                         "integers n with 0 <= n < len(children). Furthermore is must not \n"
                         "contain any duplicates.")

    # Bereite Variablen vor und verhindere damit, dass x korrumpiert wird
    dt = x.dtree
    root = 0
    ind = dt.get_right(root)
    xB = deepcopy(x.B)
    xU = deepcopy(x.U)

    # Eliminiere Matrix im Wurzelknoten
    # Reshape root tensor zu matrix
    shape = xB[root].shape
    reshaped_Broot = xB[root].reshape((shape[0], shape[1]), order="F")
    if not dt.is_leaf(ind):
        xB[ind] = mode_multiplication(reshaped_Broot, xB[ind], 2)
    else:
        xU[ind] = xU[ind] @ reshaped_Broot

    xB[root] = np.identity(xB[root].shape[0])
    # Reshapen der Wurzel
    shape = xB[root].shape
    xB[root] = xB[root].reshape((shape[0], shape[1], 1), order="F")

    # Prüfe children auf Konsistenz mit x
    subtr = np.array(sorted(get_subtree(children, 0)))
    if not len(dt.get_children()) == len(children) \
            or not np.all(subtr == np.arange(0, len(dt.get_children()))):
        raise ValueError("children is not consistent with x")
    # Prüfe dim2ind auf Konsistenz mit children
    if len(children) != 2 * len(dim2ind) - 1 \
            or np.any(children[dim2ind] != -1):
        raise ValueError("dim2ind is not consistent with children.")

    # Berechne Mapping zwischen alten und neuen Indizes
    old2new = np.zeros(dt.get_nr_nodes()).astype(int)
    new2old = np.zeros(dt.get_nr_nodes()).astype(int)

    # dt.children, dt.dim2ind, old, t
    #      children,      dim2ind, new, jj
    dt_parent = dt.get_parent()
    for t in range(len(children) - 1, 0, -1):

        if np.all(children[t] == -1):
            idx = np.argmax(dim2ind == t)
            jj = dt.get_dim2ind()[idx]

        else:
            jj_left = new2old[children[t, 0]]
            jj_right = new2old[children[t, 1]]

            par_jj_left = dt_parent[jj_left]
            par_jj_right = dt_parent[jj_right]

            if par_jj_left == par_jj_right:
                jj = par_jj_left

            elif np.any(par_jj_left == dt.children[jj_right]):
                jj = par_jj_left

            elif np.any(par_jj_right == dt.children[jj_left]):
                jj = par_jj_right

            elif par_jj_left == root:
                jj = par_jj_right

            elif par_jj_right == root:
                jj = par_jj_left

            else:
                raise ValueError("Dimensionsbäume sind nicht konsistent.")

        old2new[jj] = t
        new2old[t] = jj

    # Check mapping
    if not np.all(old2new[new2old] == np.arange(dt.get_nr_nodes())):
        raise ValueError("Dimensionsbäume sind nicht konsistent.")

    # Konstruiere U und B des neuen HTucker Tensors aus U und B des alten HTucker Tensors
    U = {}
    B = {}

    parent = -1 * np.ones(len(children)).astype(int)
    ind = np.argwhere(children[:, 0] != -1).flatten()
    parent[children[ind, 0]] = ind
    parent[children[ind, 1]] = ind

    for t in range(1, len(children)):
        jj = new2old[t]

        ii1 = children[t, 0]
        ii2 = children[t, 1]

        if ii1 == -1 and ii2 == -1:
            U[t] = xU[jj]

        else:
            iipar = parent[t]
            if iipar == 0:
                siblings = children[iipar]
                if siblings[0] != t:
                    iipar = siblings[0]
                else:
                    iipar = siblings[1]

            jj1 = dt.children[jj, 0]
            jj2 = dt.children[jj, 1]
            jjpar = dt.get_parent(jj)
            if dt.is_root(jjpar):
                siblings = dt.get_children(jjpar)
                if siblings[0] != jj:
                    jjpar = siblings[0]
                else:
                    jjpar = siblings[1]

            new_modes = new2old[[ii1, ii2, iipar]]
            old_modes = np.array([jj1, jj2, jjpar])
            permB = np.array([-1, -1, -1])

            if old_modes[0] == new_modes[0]:
                permB[0] = 0
            elif old_modes[1] == new_modes[0]:
                permB[0] = 1
            else:
                permB[0] = 2

            if old_modes[0] == new_modes[1]:
                permB[1] = 0
            elif old_modes[1] == new_modes[1]:
                permB[1] = 1
            else:
                permB[1] = 2

            if old_modes[0] == new_modes[2]:
                permB[2] = 0
            elif old_modes[1] == new_modes[2]:
                permB[2] = 1
            else:
                permB[2] = 2

            B_jj_permuted = np.moveaxis(xB[jj], source=[0, 1, 2], destination=permB)
            B[t] = np.array(B_jj_permuted)

    ind = children[0, 0]
    if ind in B.keys():
        B[0] = np.identity(B[ind].shape[2])
        # Reshape zu 3d
        shape = B[0].shape
        B[0] = B[0].reshape((shape[0], shape[1], 1), order="F")
    else:
        B[0] = np.identity(U[ind].shape[1])
        # Reshape zu 3d
        shape = B[0].shape
        B[0] = B[0].reshape((shape[0], shape[1], 1), order="F")

    new_dtree = dimtree(children=children, dim2ind=dim2ind)
    new_htucker = cls(U=U, B=B, dtree=new_dtree, is_orthog=False)
    return new_htucker


def get_subtree(children, ind):
    """
    Gibt den subtree des Knotens 'ind' definiert durch 'children' zurück.
    Helferfunktion für change_dimtree.
    """
    nodes = [ind]
    subtree = []
    while len(nodes) > 0:
        node = nodes.pop(0)
        subtree += [node]
        if children[node][0] != -1:
            nodes += list(children[node])
    return subtree
