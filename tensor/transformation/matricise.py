import numpy as np


def matricise(X, t, copy=False):
    """
    Berechnet die t-Matrizierung des Tensors 'X'.
    Falls copy==True, wird die Matrizierung mit einer Kopie von 'X' durchgeführt.
    Selbst wenn copy==False, kann es vorkommen, dass mit einer Kopie von 'X' gearbeitet wird.
    Grund dafür ist numpy.reshape, das ggf. eine Kopie erstellt.
    @param X: N-D np.ndarray mit N>=2
    @param t: Liste nicht-negativer ints
    @param copy: bool
    @return: np.ndarray
    """

    # Prüfen der Argumente
    if not isinstance(X, np.ndarray):
        raise ValueError("'X' muss ein N-D np.ndarray mit N>=2 sein.")
    if not len(X.shape) >= 2:
        raise ValueError("'X' muss ein N-D np.ndarray mit N>=2 sein.")
    if not isinstance(t, list):
        raise ValueError("'t' muss eine Liste nicht-negativer ints ohne Duplikate sein.")
    if not all(np.issubdtype(type(item), int) for item in t):
        raise ValueError("'t' muss eine Liste nicht-negativer ints ohne Duplikate sein.")
    if not all(item >= 0 for item in t):
        raise ValueError("'t' muss eine Liste nicht-negativer ints ohne Duplikate sein.")
    if not len(t) == len(set(t)):
        raise ValueError("'t' muss eine Liste nicht-negativer ints ohne Duplikate sein.")
    if not all(item < len(X.shape) for item in t):
        raise ValueError("'t' und 'X' sind nicht kompatibel.")
    if not isinstance(copy, bool):
        raise ValueError("'copy' muss vom Typ bool sein.")

    # Kopiere ggf. 'X'
    if copy:
        X = np.copy(X)

    # Ordne die Modi von 'X' so, dass die Modi aus 't' fuehrend sind
    destination = list(range(len(t)))
    X_mat = np.moveaxis(X, t, destination)

    # Berechne die Anzahl an Zeilen
    rowcount = np.array(X.shape)[np.array(t)].prod()

    # Matriziere gemaess t
    X_mat = X_mat.reshape((rowcount, -1), order='F')

    return X_mat