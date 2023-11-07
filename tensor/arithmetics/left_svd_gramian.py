import numpy as np


def left_svd_gramian(x):
    """
    Berechnet die linken Singulärvektoren und zugehörigen Singulärwerte einer Matrix A, wobei das übergebene
    Argument 'x' AA^T entspricht. Die Berechnung basiert auf einer Eigenwertzerlegung von 'x'.
    @param x: 2-D np.ndarray
    @return: (2-D np.ndarray, 1-D np.ndarray)
    """
    # Argument checks
    if not isinstance(x, np.ndarray):
        raise TypeError("'x' muss ein symmetrischer 2-D np.ndarray sein.")
    if not len(x.shape) == 2:
        raise ValueError("'x' muss ein symmetrischer 2-D np.ndarray sein.")
    if not x.shape[0] == x.shape[1]:
        raise ValueError("'x' muss ein symmetrischer 2-D np.ndarray sein.")
    if not np.allclose(x, x.T):
        raise ValueError("'x' muss ein symmetrischer 2-D np.ndarray sein.")

    # Eigenwertzerlegung
    s, u = np.linalg.eig(x)

    # Abtrennen des imaginären Anteils
    # Theoretisch sollte dieser ohnehin nicht vorhanden sein. Aufgrund von Rundungsfehlern ist 'x' jedoch nur bis
    # auf einige Nachkommastellen symmetrisch, weswegen dieser Schritt notwendig ist.
    s = np.real(s)
    u = np.real(u)

    # Singulärwerte entsprechen den Quadratwurzeln
    s = np.sqrt(np.abs(s))

    # Sortierung in absteigender Reihenfolge
    idc = np.argsort(s)[::-1]
    s = s[idc]
    u = u[:, idc]

    return u, s
