import numpy as np


def merge_modes(x, modes):
    """
    Verschmilzt die Modi von 'x' definiert in 'modes' mit Fortran-like Reihenfolge.
    @param x: ND-np.ndarray
    @param modes: list nicht-negativer ints
    @return: M-D np.ndarray with M < N
    """
    # Argument checks
    if not isinstance(x, np.ndarray):
        raise TypeError("'x' muss ein ND-np.ndarray mit N=>2 sein.")
    if not len(x.shape) >= 2:
        raise ValueError("'x' muss ein ND-np.ndarray mit N=>2 sein.")
    if not isinstance(modes, list):
        raise TypeError("'modes' muss eine list nicht-negativer ints ohne Duplikate sein.")
    if not all(np.issubdtype(type(item), np.integer) for item in modes):
        raise ValueError("'modes' muss eine list nicht-negativer ints ohne Duplikate sein.")
    if not all(item >= 0 for item in modes):
        raise ValueError("'modes' muss eine list nicht-negativer ints ohne Duplikate sein.")
    if not all(item < len(x.shape) for item in modes):
        raise ValueError("'modes' und 'x' sind nicht konsistent.")
    if not len(modes) == len(set(modes)):
        raise ValueError("'modes' muss eine list nicht-negativer ints ohne Duplikate sein.")

    # Arbeite auf einer Kopie von 'x'
    B = np.array(x)
    # Speichere alte Modusstruktur
    old_shape = B.shape
    # Ordne die Modi so um, dass die zu verschmelzenden Modi führend sind
    B = np.moveaxis(B, source=modes, destination=list(range(len(modes))))
    # Bestimme die Modusstruktur nach der Verschmelzung
    new_shape = np.array(old_shape)
    new_shape[0] = np.prod(modes)
    new_shape = np.delete(new_shape, modes)
    # Verschmelze die Modi
    B = B.reshape(new_shape, order="F")
    return B
