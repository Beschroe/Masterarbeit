import numpy as np


def get_i_max(R0, S0, sigma):
    """
    Gibt die von der Theorie des klassischen SIR-Modells vorhergesagte maximalen Prävalenz zurück.
    """
    assert sigma >= 1
    return 1 - R0 - (1 / sigma) - (np.log(sigma * S0) / sigma)
