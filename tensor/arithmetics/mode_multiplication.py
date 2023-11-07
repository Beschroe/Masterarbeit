import numpy as np


def mode_multiplication(U, A, mu):
    """
    Berechnet die Modusmultiplikation U o_{mu} A.
    @param U: 2-D np.ndarray
    @param A: N-D np.ndarray
    @param mu: nicht-negativer int
    @return: np.ndarray
    """
    # Argument checks
    if not isinstance(U, np.ndarray):
        raise TypeError("'U' muss ein 2D-np.ndarray sein.")
    if not len(U.shape) == 2:
        raise ValueError("'U' muss ein 2D-np.ndarray sein.")
    if not isinstance(A, np.ndarray):
        raise TypeError("'A' muss ein ND-np.array mit N >= 1 sein.")
    if not len(A.shape) >= 1:
        raise ValueError("'A' muss ein ND-np.array mit N >= 1 sein.")
    if not np.issubdtype(type(mu), int):
        raise TypeError("'mu' muss ein int mit 0 <= mu < len(A.shape) sein.")
    if not (0 <= mu < len(A.shape)):
        raise ValueError("'mu' muss ein int mit 0 <= mu < len(A.shape) sein.")
    if not U.shape[1] == A.shape[mu]:
        raise ValueError("U.shape[1] und A.shape[mu] müssen übereinstimmen.")

    # Multiplizieren
    product = np.tensordot(U, A, axes=(1, mu))
    # Permutieren der Modi, sodass die vorherige Reihenfolge wiederhergestellt wird
    source = list(range(mu + 1))
    destination = [mu] + list(range(mu))
    product = np.moveaxis(product, source=source, destination=destination)
    return product
