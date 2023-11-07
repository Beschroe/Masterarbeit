import numpy as np
from tensor.arithmetics.mode_multiplication import mode_multiplication
from tensor.utils.dimtree import dimtree
from copy import deepcopy


def squeeze(cls, x, dims=None, copy=True):
    """
    Entfernt Modi mit Größe 1.
    Falls 'dims' None ist, werden alle diese Modi entfernt.
    Falls 'dims' nicht None ist, werden die in 'dims' enthaltenen Modi mit Größe 1 entfernt.
    """

    # Argument Checks
    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not isinstance(dims, list) and not isinstance(dims, tuple) \
            and not isinstance(dims, np.ndarray) and dims is not None:
        raise TypeError("'dims' muss entweder None, eine list, ein tuple oder np.ndarray sein.")
    if isinstance(dims, list) or isinstance(dims, tuple):
        dims = np.array(dims)
    elif dims is None:
        dims = np.array(range(x.order))
    if not all(x.shape[i] == 1 for i in dims):
        raise ValueError("Alle in 'dims' gelisteten Modi müssen Größe 1 haben.")
    if not isinstance(copy, bool):
        raise TypeError("'copy' muss vom Typ bool sein.")

    # Vorbereitung der Variablen des neuen hierarchischen Tuckertensors
    # Dazu werden ggf. einige Variablen des alten hierarchischen Tuckertensors kopiert, um ein Korrumpieren eben jenes
    # hierarchischen Tuckertensors zu verhindern
    dtree = x.dtree.copy()
    children = dtree.get_children()
    dim2ind = dtree.get_dim2ind()
    parent = dtree.get_parent()
    nr_nodes = dtree.get_nr_nodes()
    shape = np.array(x.shape)
    is_orthog = x.is_orthog
    if copy:
        xU = deepcopy(x.U)
        xB = deepcopy(x.B)
    else:
        xU = x.U
        xB = x.B

    to_squeeze = np.array([False] * dtree.get_nr_nodes())
    to_squeeze[dims] = True

    if np.all(np.array(x.shape) == 1) and len(dims) == x.order:
        # Spezialfall: Nach dem Entfernen der Modi mit Größe 1 bleibt nur ein Skalar übrig
        squeezed = x.full()
        return squeezed

    if x.order == 2:
        # Spezialfall: hierarchische Tuckertensoren mit nur zwei Modi bleiben unberührt
        squeezed = x
        return squeezed

    while len(np.where(shape > -1)[0]) > 2:
        # Index des nächsten Blattknotens, der eliminiert wird
        ind = next_single_node(dtree, shape, to_squeeze)

        if ind == -1:
            # keine zu eliminierenden Knoten sind übrig
            break
        ind_sibling = dtree.get_sibling(ind)
        ind_par = dtree.get_parent(ind)

        # Der zu ind gehörige Modus
        d_ind = dtree.get_dim(ind)
        assert len(d_ind) == 1
        d_ind = d_ind[0]

        # Wende xU[ind] auf den Transfertensor xB des Elternknotens an und erhalte die Matrix tmp
        if dtree.is_left(ind):
            tmp = xU[ind] * xB[ind_par]
            tmp = np.sum(tmp, axis=0)
        else:
            tmp = xU[ind] * xB[ind_par]
            tmp = np.sum(tmp, axis=1)

        if dtree.is_leaf(ind_sibling):
            # Geschwisterknoten ist auch ein Blatt
            # Wende die Blattmatrix des Geschwisterknotens auf tmp an
            xU[ind_par] = xU[ind_sibling] @ tmp

            # Setze den vorherigen Transfertensor auf 0
            xB[ind_par] = np.zeros((0, 0, 0))

            # Update die Struktur des Dimensionsbaums
            children[ind_par] = [-1, -1]
            parent[ind] = -1
            parent[ind_sibling] = -1
            shape[d_ind] = -1
            dim2ind[np.where(dim2ind == ind_sibling)] = ind_par

        else:
            # Geschwisterknoten ist ein innerer Knoten
            # Wende die Matrix tmp auf den Transfertensor des Geschwisterknotens an, um damit den neuen Transfertensor
            # des Elternknotens zu erhalten
            xB[ind_par] = mode_multiplication(tmp.T, xB[ind_sibling], 2)

            # Die Knoten ind und ind_sibling werden jetzt nicht mehr genutzt
            # Die Nachfahren von ind_sibling sind jetzt mit ind_par verbunden
            children[ind_par] = children[ind_sibling]
            parent[children[ind_par][0]] = ind_par
            parent[children[ind_par][1]] = ind_par

            # Markiere d_ind als reduziert
            shape[d_ind] = -1

    # Eliminiere nun die ungenutzten Knoten
    # 1) Die Indizes der noch genutzten Knoten
    new2old_nodes = np.array(get_subtree(children, 0))
    # 2) Andere Richtung: Die neuen Kontenindizes der alten Knoten
    old2new_nodes = -1 * np.ones(nr_nodes).astype(int)
    old2new_nodes[new2old_nodes] = np.array(range(len(new2old_nodes)))
    # 3) Eliminiere die ungenutzten Knoten
    children = children[new2old_nodes]
    lasting_dims = np.where(shape > -1)
    dim2ind = old2new_nodes[dim2ind[lasting_dims]]
    xB = {old2new_nodes[k]: xB[k] for k in new2old_nodes if k in xB.keys()}
    xU = {old2new_nodes[k]: xU[k] for k in new2old_nodes if k in xU.keys()}
    # 4) Aktualisiere die Knotenindizes in children
    children[np.where(children > 0)] = old2new_nodes[children[np.where(children > 0)]]

    # Erzeuge den resultierenden hierarchischen Tuckertensor
    dtree = dimtree(children=children, dim2ind=dim2ind)
    new_htucker = cls(dtree=dtree, U=xU, B=xB, is_orthog=is_orthog)
    return new_htucker


def next_single_node(dtree, shape, to_squeeze):
    """
    Helferfunktion: Gibt die Knoten im Dimensionsbaum 'dtree' zurück, deren Größe 1 ist. Falls zwei benachbarte
    Knoten Größe 1 haben, werden diese zuerst zurückgegeben.
    """

    dim2ind = dtree.get_dim2ind()
    children = dtree.get_children()
    to_squeeze = np.array(to_squeeze)

    # All Blattknoten deren Modusgröße 1 ist
    shape_ind = list(np.where(shape == 1)[0])
    squeeze_ind = list(np.where(to_squeeze)[0])
    single_dims = np.array([i for i in shape_ind if i in squeeze_ind])
    if len(single_dims) == 0:
        # Keine derartigen Blattknoten vorhanden
        return -1
    single_nodes = dim2ind[single_dims]

    # Elternknoten deren beide Kinder jeweils Modusgröße 1 haben
    single_node_par = [dtree.get_parent(ind) for ind in list(single_nodes)]
    par_two_single_nodes = [ind for ind in single_node_par if single_node_par.count(ind) == 2]
    par_two_single_nodes = np.array(list(set(par_two_single_nodes))).astype(int)  # Entferne Duplikate
    sibling_singletons = children[par_two_single_nodes]

    if len(sibling_singletons) > 0:
        return sibling_singletons[0][0]
    elif len(single_nodes) > 0:
        return single_nodes[0]
    else:
        return -1


def get_subtree(children, ind):
    """
    Helferfunktion: Gibt basierend auf 'children' den subtree zurück, dessen Wurzel dem durch 'ind' referenzierten
    Knoten entspricht.
    """
    nodes = [ind]
    subtree = []
    while len(nodes) > 0:
        node = nodes.pop(0)
        subtree += [node]
        if children[node][0] != -1:
            nodes += list(children[node])
    return subtree
