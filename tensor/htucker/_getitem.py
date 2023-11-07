import numpy as np
from copy import deepcopy


def get(self, key):
    """
    Falls key ein einziges Element referenziert, wird dieses Element als float zurückgegeben.
    Falls key mehr als ein Element referenziert, wird ein neuer hierarchischer Tuckertensor, der diese Elemente umfasst,
    zurückgegeben.
    @param key: tuple or slice or int
    @return: float or tensor.htucker.tensor
    """
    if isinstance(key, slice):
        raise ValueError("Der hierarchische Tuckertensor umfasst {} Dimensions. Für jede davon muss ein index/slice"
                         " angegeben werden.".format(self.order))

    elif np.issubdtype(type(key), np.integer):
        raise ValueError("Der hierarchische Tuckertensor umfasst {} Dimensions. Für jede davon muss ein index/slice"
                         " angegeben werden.".format(self.order))

    elif isinstance(key, tuple):
        if not len(key) == self.order:
            raise ValueError("Der hierarchische Tuckertensor umfasst {} Dimensions. Für jede davon muss ein index/slice"
                             " angegeben werden.".format(self.order))

        # Kopiere den zugrundeliegenden hierarchischen Tuckertensor, um ein Korrumpieren zu verhindern
        z = deepcopy(self)

        ind = key
        for t in z.dtree.get_leaves():
            # Schränke die Blattmatrizen in t entsrechend ind[dim_t] ein
            dim_t = z.dtree.get_dim(t)[0]
            ind_t = ind[dim_t]

            if np.issubdtype(type(ind_t), np.integer):
                check_index(ind_t, dim_t, z.shape)
                z.U[t] = z.U[t][ind_t, :].reshape((1, -1))
            elif isinstance(ind_t, slice):
                start, stop, step = ind_t.indices(z.shape[dim_t])
                check_slice(start, stop, step, dim_t, z.shape)
                z.U[t] = z.U[t][start:stop:step, :]
            else:
                raise ValueError("Jeder index/slice muss ein nicht-negativer int sein bzw. aus solchen bestehen.")

        # Update shape von z
        z.update_shape()
        if all(i == 1 for i in z.shape):
            # Return float
            return z.full().reshape(1)[0]
        else:
            return z

    else:
        raise ValueError("Der hierarchische Tuckertensor umfasst {} Dimensions. Für jede davon muss ein index/slice"
                             " angegeben werden.".format(self.order))


def check_index(ind, dim, shape):
    """
    Helferfunktion, zum Prüfen eines einzelnen Index.
    """
    if not ind >= 0:
        raise ValueError("Ungültiger Index. Jeder Index muss ein nicht-negativer int sein.")
    if ind >= shape[dim]:
        raise ValueError("Ungültiger Index. Index {} zu groß für Modus {} der Größe {}.".format(ind, dim, shape[dim]))
    return


def check_slice(start, stop, step, dim, shape):
    """
    Helferfunktion, zum Prüfen eines Slice.
    """
    if start < 0 or stop < 0 or step < 0:
        raise ValueError("Ungültiger Slice. Jeder Slice muss aus nicht-negativen ints bestehen.")
    if start >= stop:
        raise ValueError("Ungültiger Slice. Start Parameter {} => Sop Parameter {}.".format(start, stop))
    if start >= shape[dim]:
        raise ValueError("Ungültiger Slice. Start parameter {} ist zu groß für Modus {} mit "
                         "Größe {}.".format(start, dim, shape[dim]))
    if stop > shape[dim]:
        raise ValueError("Ungültiger Slice. Stop parameter {} ist zu groß für Modus {} mit "
                         "Größe {}.".format(start, dim, shape[dim]))
    return
