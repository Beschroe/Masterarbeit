import numpy as np


def multi_mul(x, U, modes):
    """
    Berechnet die multilineare Multiplikation U o x
    @param x: ND-np.ndarray
    @param U: list of 2D-np.ndarrays
    @param modes: list nicht-negativer ints
    @return: ND-np.ndarray
    """
    # Argument checks
    if not isinstance(x, np.ndarray):
        raise ValueError("'x' muss ein ND-np.ndarray mit N >= 1 sein.")
    if not len(x.shape) >= 1:
        raise ValueError("'x' muss ein ND-np.ndarray mit N >= 1 sein.")
    if not isinstance(U, list):
        raise ValueError("'U' muss eine list bestehnd aus 2D-np.ndarrays sein.")
    if not all(isinstance(item, np.ndarray) for item in U):
        raise ValueError("'U' muss eine list bestehnd aus 2D-np.ndarrays sein.")
    if not all(len(item.shape) == 2 for item in U):
        raise ValueError("'U' muss eine list bestehnd aus 2D-np.ndarrays sein.")
    if not isinstance(modes, list):
        raise ValueError("'modes' muss eine list nicht-negativer ints ohne Duplikate sein.")
    if not all(np.issubdtype(type(item), np.integer) for item in modes):
        raise ValueError("'modes' muss eine list nicht-negativer ints ohne Duplikate sein.")
    if not all(item >= 0 for item in modes):
        raise ValueError("'modes' muss eine list nicht-negativer ints ohne Duplikate sein.")
    if not len(modes) == len(set(modes)):
        raise ValueError("'modes' muss eine list nicht-negativer ints ohne Duplikate sein.")
    if not all(item < len(x.shape) for item in modes):
        raise ValueError("'modes' und 'x' sind nicht kompatibel.")
    if not all(U[item].shape[1] == x.shape[item] for item in modes):
        raise ValueError("'x', 'U' und 'modes' sind nicht kompativel.")

    product = None
    for i in range(len(U)):
        if product is None:
            product = np.tensordot(x, U[i], axes=[modes[i], 1])
        else:
            product = np.tensordot(product, U[i], axes=[modes[i], 1])
        # Wiederherstellung der vorherigen Modusreihenfolge mit neuen Modusgrößen
        source = list(range(len(product.shape)))
        destination = list(range(modes[i])) + list(range(modes[i] + 1, len(product.shape) + 1))
        destination[len(destination)-1] = modes[i]
        product = np.moveaxis(product, source=source, destination=destination)
    return product




