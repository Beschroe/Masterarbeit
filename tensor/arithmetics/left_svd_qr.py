import numpy as np


def left_svd_qr(x):
    """
    Berechnet die linken Singulärvektoren samt zugehöriger Singulärwerte von 'x'.
    Die Berechnung basiert auf einer QR-Zerlegung mit anschließender Singulärwertzerlegung.
    @param x: 2-D np.ndarray
    @return: (2-D np.ndarray, 1-D np.ndarray)
    """
    # Argument checks
    if not isinstance(x, np.ndarray):
        raise ValueError("'x' muss ein 2D np.ndarray sein")
    if len(x.shape) != 2:
        raise ValueError("'x' muss ein 2D np.ndarray sein")

    if x.shape[0] > x.shape[1]:
        q, r = np.linalg.qr(x, mode="reduced")
        u, s, _ = np.linalg.svd(r, full_matrices=False)
        u = q @ u
    else:
        _, r = np.linalg.qr(x.T, mode="reduced")
        u, s, _ = np.linalg.svd(r.T, full_matrices=False)

    return u, s
