import numpy as np
from tensor.utils.dimtree import dimtree
from copy import deepcopy


def change_root(cls, x, ind, lr_subtree="right"):
    """
    Ändert die Wurzel des Dimensionsbaums einer Kopie des hierarchischen Tuckertensors 'x'.
    Dabei gibt 'lr_subtree' an, ob 'ind' linkes oder rechtes Kind des Wurzelknotens wird.
    Der restliche Dimensionsbaum wird entsprechend angepasst.
    @param x: tensor.htucker.htucker
    @param ind: non-negative integer
    @param lr_subtree: str in ['left', 'right']
    @return: tensor.htucker.htucker
    """

    if not isinstance(x, cls):
        raise TypeError("'x' ist kein hierarchischer Tuckertensor.")
    if not np.issubdtype(type(ind), int):
        raise TypeError("'ind' ist kein nicht-negativer int zwischen 0 und 2 * x.order - 1.")
    if not ind >= 0 or ind >= 2 * x.order - 1:
        raise ValueError("'ind' ist kein nicht-negativer int zwischen 0 und 2 * x.order - 1.")
    if not isinstance(lr_subtree, str):
        raise TypeError("'lr_subtree' muss entweder 'left' oder 'right' lauten.")
    if lr_subtree not in ["left", "right"]:
        raise ValueError("'lr_subtree' muss entweder 'left' oder 'right' lauten.")

    # Vorbereitung der Variablen
    # Einige Variablen werden kopiert, um ein Korrumpieren von x zu vermeiden
    dt = x.dtree.copy()
    if ind > 0:
        # copy children list
        children = dt.get_children()
        # copy parents list
        ind_par = dt.get_parent(ind)

        if dt.is_root(ind_par):
            # Falls notwendig,
            if lr_subtree == "left":
                if dt.get_right(ind_par) == ind:
                    children[ind_par][[0, 1]] = children[ind_par][[1, 0]]
            else:
                if dt.get_left(ind_par) == ind:
                    children[ind_par][[0, 1]] = children[ind_par][[1, 0]]

        else:
            # Die neuen Kinder der Wurzel sind durch ind und ind_par gegeben
            root_children = np.array(dt.get_children(0))
            if lr_subtree == "left":
                children[0, 0] = ind
                children[0, 1] = ind_par
            else:
                children[0, 0] = ind_par
                children[0, 1] = ind

            # Vertausche die Eltern-Kind Relation der Nachfahren von ind
            ii = ind
            ii_par = ind_par
            lr = 1
            while not dt.is_root(ii_par):
                lr = 1
                if dt.is_left(ii):
                    lr = 0
                ii = ii_par
                ii_par = dt.get_parent(ii)
                children[ii, lr] = ii_par

            # Verbinde children wieder mit dem originalen Wurzelknoten
            if root_children[0] == ii:
                children[ii, lr] = root_children[1]
            else:
                children[ii, lr] = root_children[0]

        # Sortiere die Indizes so, dass 0,...nr_nodes-1 top down durch den Baum läuft
        new2old = np.zeros(dt.get_nr_nodes()).astype(int)
        ii_read = 0
        ii_write = 1

        while ii_write < dt.get_nr_nodes() - 1:
            if np.all(children[new2old[ii_read]] != [-1, -1]):
                new2old[ii_write:ii_write+2] = children[new2old[ii_read]]
                ii_write += 2
            ii_read += 1

        # Konstruiere das Mapping von alten zu neuen Indizes
        old2new = np.ones(dt.get_nr_nodes()).astype(int)
        old2new[new2old] = np.arange(dt.get_nr_nodes()).astype(int)

        # Konstruiere dim2nd und children des neuen hierarchischen Tuckertensors
        children = children[new2old]
        for ii in range(len(children)):
            if np.all(children[ii] != [-1, -1]):
                children[ii] = old2new[children[ii]]

        dim2ind = old2new[dt.get_dim2ind()]
        res = cls.change_dimtree(x, children, dim2ind)
        return res

    else:
        # Damit der ursprüngliche Wurzelknoten zum Kind werden kann, muss ein neues Level über
        # dem ursprünglichen Wurzelknoten erzeugt werden. Entsprechend gibt es eine neue Wurzel
        dt = x.dtree.copy()
        children = dt.get_children()
        ind = np.where(children > -1)
        children[ind] = children[ind] + 2

        if lr_subtree == "left":
            new_level = np.array([[2, 1], [-1, -1]])
            children = np.vstack((new_level, children))
        else:
            new_level = np.array([[1, 2], [-1, -1]])
            children = np.vstack((new_level, children))

        dim2ind = dt.get_dim2ind() + 2
        new_dim = np.array([1])
        dim2ind = np.hstack((new_dim, dim2ind))

        # Kopieren von x.U und x.B
        xU = deepcopy(x.U)
        xB = deepcopy(x.B)
        U = {(k+2): v for k, v in xU.items()}
        U[1] = np.array([1]).reshape((1, 1))

        B = {(k+2): v for k, v in xB.items()}
        B[0] = np.array([1]).reshape((1, 1, 1))

        new_dtree = dimtree(children=children, dim2ind=dim2ind)
        new_htucker = cls(U=U, B=B, dtree=new_dtree, is_orthog=False)
        return new_htucker
