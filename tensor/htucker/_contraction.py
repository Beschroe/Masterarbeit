import numpy as np
from tensor.utils.dimtree import dimtree
from tensor.arithmetics.mode_multiplication import mode_multiplication
from copy import deepcopy


def contract(cls, x, y, dims_x, dims_y):
    """
    Kontrahiert die Dimensionen 'dims_x' und 'dims_y' der beiden hierarchischen Tuckertensoren 'x' und 'y'.
    @param x: tensor.htucker.htucker
    @param y: tensor.htucker.htucker
    @param dims_x: None, list, tuple or np.ndarray
    @param dims_y: None, list, tuple or np.ndarray
    @return: tensor.htucker.htucker
    """
    # Argument checks and preparation
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not isinstance(y, cls):
        raise TypeError("'y' ist kein hierarchischer Tuckertensor.")
    if dims_x is not None and not isinstance(dims_x, list) \
            and not isinstance(dims_x, tuple) and not isinstance(dims_x, np.ndarray):
        raise ValueError("'dims_x' müssen 'dims_y' vom Typ None, list, tuple oder numpy.ndarray sein.")
    if dims_y is not None and not isinstance(dims_y, list) \
            and not isinstance(dims_y, tuple) and not isinstance(dims_y, np.ndarray):
        raise ValueError("'dims_x' müssen 'dims_y' vom Typ None, list, tuple oder numpy.ndarray sein.")
    if dims_x is None and dims_y is None:
        dims_x = np.array([]).astype(int)
        dims_y = np.array([]).astype(int)
    if dims_y is None:
        dims_y = np.array(dims_x).astype(int)
    else:
        if not all(np.issubdtype(type(n), int) for n in dims_y) or not all(n >= 0 for n in dims_y):
            raise ValueError("Alle Elemente aus 'dims_x' und 'dims_y' müssen positive ints sein. Außerdem dürfen keine"
                             "Duplikate vorhanden sein.")
        if len(set(dims_x)) < len(dims_x):
            raise ValueError("Alle Elemente aus 'dims_x' und 'dims_y' müssen positive ints sein. Außerdem dürfen keine"
                             "Duplikate vorhanden sein.")
    if dims_x is None:
        dims_x = np.array(dims_y).astype(int)
    else:
        if not all(np.issubdtype(type(n), int) for n in dims_x) or not all(n >= 0 for n in dims_x):
            raise ValueError("Alle Elemente aus 'dims_x' und 'dims_y' müssen positive ints sein. Außerdem dürfen keine"
                             "Duplikate vorhanden sein.")
        if len(set(dims_x)) < len(dims_x):
            raise ValueError("Alle Elemente aus 'dims_x' und 'dims_y' müssen positive ints sein. Außerdem dürfen keine"
                             "Duplikate vorhanden sein.")

    if not np.all(np.array(x.shape)[dims_x] == np.array(y.shape)[dims_y]):
        raise ValueError("Die zu kontrahierenden Dimensionen von 'x' und 'y' müssen übereinstimmen.")

    # Herausfinden, welcher Kontraktionsfall gegeben ist
    compl_dims_x = set(range(x.order)) - set(dims_x)
    compl_dims_x = np.array(sorted(list(compl_dims_x))).astype(int)

    compl_dims_y = set(range(y.order)) - set(dims_y)
    compl_dims_y = np.array(sorted(list(compl_dims_y))).astype(int)

    roots_x = get_subtree_by_dims(x.dtree, dims_x)
    compl_roots_x = get_subtree_by_dims(x.dtree, compl_dims_x)

    roots_y = get_subtree_by_dims(y.dtree, dims_y)
    compl_roots_y = get_subtree_by_dims(y.dtree, compl_dims_y)

    if 1 in [len(roots_x), len(compl_roots_x)] and \
            1 in [len(roots_y), len(compl_roots_y)]:
        # Fall: One ode
        compl_x = False
        compl_y = False
        if len(roots_x) == 1:
            node_x = roots_x[0]
        else:
            node_x = compl_roots_x[0]
            compl_x = True
        if len(roots_y) == 1:
            node_y = roots_y[0]
        else:
            node_y = compl_roots_y[0]
            compl_y = True

        contracted_tensor = one_node(cls, x, y, dims_x, dims_y, node_x, node_y, compl_x, compl_y)
        return contracted_tensor

    elif len(dims_x) == len(x.shape) or len(dims_y) == len(y.shape):
        # Fall zwei Knoten ist möglich
        # Falls es nützlich ist, werden 'x' und 'y' geswapped
        if len(dims_y) != len(y.shape):
            x, y = y, x
            dims_x, dims_y = dims_y, dims_x
            roots_x, roots_y = roots_y, roots_x
            compl_roots_x, compl_roots_y = compl_roots_y, compl_roots_x

        dtx = x.dtree
        # Prüfe, ob zwei subtrees in 'x' vorhanden sind
        # Entweder die beiden Wurzeln dieser subtrees teilen eine Eltern-Kind Relation oder
        # sind beide sind Kinder der Wurzel
        node_x_child = -7
        if len(roots_x) == 2:
            nodes = sorted([dtx.get_parent(roots_x[0]), dtx.get_parent(roots_x[1])])
            if dtx.get_parent(nodes[1]) == nodes[0] \
                    or (dtx.get_parent(nodes[0]) == 0
                        and dtx.get_parent(nodes[1]) == 0):
                node_x_child = nodes[1]

        # Prüfe auf Wurzeln in compl_dims_x:
        # Entweder die beiden Wurzeln dieser subtrees teilen eine Eltern-Kind Relation oder
        # sind beide sind Kinder der Wurzel
        if node_x_child == -7 and len(compl_roots_x) == 2:
            nodes = sorted([dtx.get_parent(compl_roots_x[0]), dtx.get_parent(compl_roots_x[1])])
            if dtx.get_parent(nodes[1]) == nodes[0] \
                    or (dtx.get_parent(nodes[0]) == 0
                        and dtx.get_parent(nodes[1]) == 0):
                node_x_child = nodes[1]

        if node_x_child == -7:
            raise ValueError("Die Kontraktion kann für diesen Fall nicht im hierarchischen Tuckerformat berechnet "
                             "werden.")

        contracted_tensor = two_nodes(cls, x, y, dims_x, dims_y, node_x_child)
        return contracted_tensor
    else:
        raise ValueError("Die Kontraktion kann für diesen Fall nicht im hierarchischen Tuckerformat berechnet "
                         "werden.")


# Helferfunktionen

def two_nodes(cls, x, y, dims_x, dims_y, node_x_child):
    """
    Berechnet die Kontraktion für 'x', 'y', 'dims_x' und 'dims_y' für den Fall, dass 'dims_y'=len('y'.shape) gilt und
    dass 'dims_x' durch zwei Knoten abgebildet werden kann, die durch eine Kante verbunden sind.
    """
    dims_x = np.array(dims_x)
    dims_y = np.array(dims_y)

    # Tausche den Wurzelknoten aus
    x_hat = cls.change_root(x, node_x_child, lr_subtree="right")

    dtx = x_hat.dtree

    ii_left = dtx.get_left(0)
    ii_right = dtx.get_right(0)

    # Berechne lr_left, lr_right so, dass
    # lr_left indicates which child of left contains some of dims_x,
    # lr_right indicates which child of right contains some of dims_x.

    # Teile 'dims_x' in zwei subtrees dims_x_left, dims_x_right
    ii = ii_left
    while not dtx.is_leaf(ii):
        ii = dtx.get_left(ii)
    if np.any(dtx.get_dim(ii) == dims_x):
        dims_x_left = dtx.get_dim(dtx.get_left(ii_left))
        lr_left = 0
    else:
        dims_x_left = dtx.get_dim(dtx.get_right(ii_left))
        lr_left = 1
    ii = ii_right
    while not dtx.is_leaf(ii):
        ii = dtx.get_left(ii)
    if np.any(dtx.get_dim(ii) == dims_x):
        dims_x_right = dtx.get_dim(dtx.get_left(ii_right))
        lr_right = 0
    else:
        dims_x_right = dtx.get_dim(dtx.get_right(ii_right))
        lr_right = 1

    # Teile 'dims_y' in zwei subtrees dims_y_left, dims_y_right
    dims_y_left = np.zeros(len(dims_x_left)).astype(int)
    for i in range(len(dims_x_left)):
        dims_y_left[i] = dims_y[np.where(dims_x == dims_x_left[i])]
    dims_y_right = np.zeros(len(dims_x_right)).astype(int)
    for i in range(len(dims_x_right)):
        dims_y_right[i] = dims_y[np.where(dims_x == dims_x_right[i])]

    roots_y_left = get_subtree_by_dims(y.dtree, dims_y_left)
    roots_y_right = get_subtree_by_dims(y.dtree, dims_y_right)

    if len(roots_y_left) == 1:
        y_hat = cls.change_root(y, roots_y_left[0], "left")
    elif len(roots_y_right) == 1:
        y_hat = cls.change_root(y, roots_y_right[0], "right")
    else:
        raise ValueError("Kontraktion kann für diesen Fall nicht im hierarchischen Tuckerformat berechnet werden.")

    # Bestimme Eliminationsmatrizen
    x_left = cls.change_root(x_hat, dtx.get_children(ii_left)[lr_left], "right")
    M_left = elim_matrix(x_left, y_hat, dims_x_left, dims_y_left)
    x_right = cls.change_root(x_hat, dtx.get_children(ii_right)[lr_right], "right")
    M_right = elim_matrix(x_right, y_hat, dims_x_right, dims_y_right)
    # Kombiniere die Matrizen zur Gesamtmatrix
    B_as_mat = y_hat.B[0].reshape((y_hat.B[0].shape[0], y_hat.B[0].shape[1]), order="F")
    My = M_left @ B_as_mat @ M_right.T

    B0 = x_hat.B[0].reshape((x_hat.B[0].shape[0], x_hat.B[0].shape[1]))
    B_ = mode_multiplication(B0, x_hat.B[ii_right], 2)
    B_ = mode_multiplication(My, B_, lr_right)

    M = np.tensordot(x_hat.B[ii_left], B_, axes=[[2, lr_left], [2, lr_right]])

    # Verbinde die subtrees mit der Wurzel
    # Transfertensor der Wurzel ist die Eliminationsmatrix M
    x_hat.B[0] = M.reshape((M.shape[0], M.shape[1], 1))
    children = deepcopy(dtx.get_children())
    children[0, 0] = dtx.get_children()[ii_left][1 - lr_left]
    children[0, 1] = dtx.get_children()[ii_right][1 - lr_right]
    dim2ind = deepcopy(dtx.get_dim2ind())

    # Eliminiere die ungenutzten Knoten und passe die Modusnummerierung an
    new_htucker = adjust_tree(cls, children, dim2ind, x_hat.U, x_hat.B, (x_hat.is_orthog and y_hat.is_orthog))
    return new_htucker


def one_node(cls, x, y, dims_x, dims_y, node_x, node_y, compl_x, compl_y):
    """
    Berechnet die Kontraktion für 'x', 'y', 'dims_x' und 'dims_y' für den Fall, dass
    compl_x == False: subtree von node_x beinhaltet alle Modi aus 'dims_x'
    compl_x == True: subtree von node_x beinhaltet das Komplement der Modi aus 'dims_x'
    compl_y == False: subtree von node_y beinhaltet alle Modi aus 'dims_y'
    compl_y == True: subtree von node_y beinhaltet das Komplement der Modi aus 'dims_y'
    für 'x' und 'y' gilt.
    """
    # Austausch der Wurzel, sodass der rechte subtree alle Modi aus 'dims_x' enthält
    if not compl_x:
        x_hat = cls.change_root(x, ind=node_x, lr_subtree="right")
    else:
        x_hat = cls.change_root(x, ind=node_x, lr_subtree="left")

    # Das Gleiche für 'y'
    if not compl_y:
        y_hat = cls.change_root(y, ind=node_y, lr_subtree="right")
    else:
        y_hat = cls.change_root(y, ind=node_y, lr_subtree="left")

    # Falls entweder alle Knoten oder kein Knoten ausgewählt wurden, ist node_x gleich 0. Ob dies der Fall ist, wird von
    # compl_x angezeigt. Ist also node_x gleich 0, so fügt change_root einen neuen Knoten ein, was dazu führt, dass die
    # Indizes aller weiteren Knoten um 1 verschoben werden und der neue 0 Modus die Größe 1 hat. An diese Tatsache
    # wird 'dims_x' angepasst.
    squeeze_left = False
    if node_x == 0:
        if compl_x:
            dims_x = 0
        else:
            dims_x = [item + 1 for item in dims_x]
            squeeze_left = True

    # Das Gleiche wird erneut für 'y' durchgeführt
    squeeze_right = False
    if node_y == 0:
        if compl_y:
            dims_y = 0
        else:
            dims_y = [item + 1 for item in dims_y]
            squeeze_right = True

    M = elim_matrix(x_hat, y_hat, dims_x, dims_y)

    # Kombiniere 'x' und 'y' in einen großen Baum
    offset_x = 1
    offset_y = x_hat.dtree.get_nr_nodes() + 1

    new_root_x = x_hat.dtree.get_left(0)
    new_root_y = y_hat.dtree.get_left(0)

    # Konstruiere den kombinierten children array des neuen Dimensionsbaums
    ind = np.where(x_hat.dtree.get_children() > -1)
    xchildren = np.array(x_hat.dtree.get_children())
    xchildren[ind] = xchildren[ind] + offset_x
    ind = np.where(y_hat.dtree.get_children() > 1 - 1)
    ychildren = np.array(y_hat.dtree.get_children())
    ychildren[ind] = ychildren[ind] + offset_y
    children = [[offset_x + new_root_x, offset_y + new_root_y]] + list(xchildren) + list(ychildren)
    children = np.array(children)

    # Konstruiere die kombinierten Transfertensoren
    B = {0: M.reshape((M.shape[0], M.shape[1], 1), order="F")}
    Bx = {(k + offset_x): v for k, v in x_hat.B.items()}
    By = {(k + offset_y): v for k, v in y_hat.B.items()}
    B.update(Bx)
    B.update(By)

    # Konstruiere die kombinierten Blattmatrizen
    U = {}
    Ux = {(k + offset_x): v for k, v in x_hat.U.items()}
    Uy = {(k + offset_y): v for k, v in y_hat.U.items()}
    U.update(Ux)
    U.update(Uy)

    # Konstruiere die kombinierte Dimensionen
    dim2ind = list(x_hat.dtree.get_dim2ind() + offset_x) + list(y_hat.dtree.get_dim2ind() + offset_y)
    dim2ind = np.array(dim2ind)

    # Eliminiere alle ungenutzten Knoten und passe die Modusnummerierung entsprechend an
    prod = adjust_tree(cls, children, dim2ind, U, B, x_hat.is_orthog and y_hat.is_orthog)

    if squeeze_left and squeeze_right:
        prod = cls.squeeze(prod, copy=False)
    elif squeeze_left:
        prod = cls.squeeze(prod, prod.dtree.get_dim(prod.dtree.get_left(0)), copy=False)
    elif squeeze_right:
        prod = cls.squeeze(prod, prod.dtree.get_dim(prod.dtree.get_right(0)), copy=False)
    return prod


def get_subtree_by_dims(dtree, dims):
    """
    Helferfunktion: Bestimmt die minimal notwendigen Knoten, die notwendig sind, um die Dimensionen in 'dims' zu
    repräsentieren.
    """
    v = np.array([False] * dtree.get_nr_nodes())

    v[dtree.get_dim2ind()[dims]] = True

    for ii in range(dtree.get_nr_nodes() - 1, 0, -1):
        sibling = dtree.get_sibling(ii)
        if v[ii] and v[sibling]:
            parent = dtree.get_parent(ii)
            v[parent] = True

    roots = []

    for ii in range(dtree.get_nr_nodes() - 1, 0, -1):
        parent = dtree.get_parent(ii)
        if v[ii] and not v[parent]:
            roots += [ii]

    if v[0]:
        roots += [0]

    return roots


def adjust_tree(cls, children, dim2ind, U, B, is_orthog):
    """
    Helferfunktion: Elimiert alle ungenutzten Knoten des subtrees von 0. Die übrigbleibenden Modi werden 1,...,d
    benannt.
    """

    xU = U
    xB = B

    dtree = dimtree(children=children, dim2ind=dim2ind)
    new2old = np.array((dtree.get_subtree(0)))

    old2new = -1 * np.ones(dtree.get_nr_nodes()).astype(int)
    old2new[new2old] = list(range(len(new2old)))

    children = children[new2old]

    no_leaf = np.where(children != [-1, -1])
    children[no_leaf] = old2new[children[no_leaf]]

    xU = {old2new[i]: xU[i] for i in new2old if i in xU.keys()}
    xB = {old2new[i]: xB[i] for i in new2old if i in xB.keys()}

    dim2ind = old2new[dim2ind]
    dim2ind = dim2ind[dim2ind != -1]

    dtree = dimtree(children=children, dim2ind=dim2ind)
    new_htucker = cls(U=xU, B=xB, dtree=dtree, is_orthog=is_orthog)
    return new_htucker


def elim_matrix(x, y, dims_x, dims_y):
    """
    Helferfunktion: Bestimmt die Eliminationsmatrix des rechten subtrees von 'x' und des zugehörigen subtrees von 'y'
    mit dem dieser verbunden ist.
    """

    # Initialisiere Mappings zwischen den Knoten von 'x' und 'y'
    ix_leaves = x.dtree.get_dim2ind()[dims_x]
    iy_leaves = y.dtree.get_dim2ind()[dims_y]

    ix2iy = np.zeros(x.dtree.get_nr_nodes()).astype(int)
    iy2ix = np.zeros(y.dtree.get_nr_nodes()).astype(int)

    ix2iy[ix_leaves] = iy_leaves
    iy2ix[iy_leaves] = ix_leaves

    # Indizes des rechten subtrees von 'x'
    root_x = 0
    root_x_right = x.dtree.get_right(root_x)
    inds_x = x.dtree.get_subtree(root_x_right)

    M = {}

    # Traversiere botttom up
    for ix in inds_x[::-1]:
        if x.dtree.is_leaf(ix):
            iy = ix2iy[ix]
            # M_t = U1_t.T @ U2_t
            M[ix] = x.U[ix].T @ y.U[iy]

        else:
            ix_left = x.dtree.get_left(ix)
            ix_right = x.dtree.get_right(ix)

            iy_left = ix2iy[ix_left]
            iy_right = ix2iy[ix_right]

            if np.all(y.dtree.get_parent(iy_left) != y.dtree.get_parent(iy_right)):
                raise ValueError("Dimensionsbäume von 'x' und 'y' sind inkompatibel.")

            iy = y.dtree.get_parent(iy_left)

            ix2iy[ix] = iy
            iy2ix[iy] = ix

            if y.dtree.is_left(iy_left):
                M_t = np.tensordot(M[ix_left], y.B[iy], axes=[1, 0])
            else:
                M_t = np.tensordot(M[ix_left], y.B[iy], axes=[1, 1])
            M_t = np.tensordot(M[ix_right], M_t, axes=[1, 1])
            M_t = np.tensordot(x.B[ix], M_t, axes=[[0, 1], [1, 0]])
            M[ix] = M_t

            del M[ix_left]
            del M[ix_right]

    return M[x.dtree.get_right(0)]
