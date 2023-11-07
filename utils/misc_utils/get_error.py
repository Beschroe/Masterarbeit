import numpy as np


def get_error(err):
    """
    Gibt auf Grundlage der knotenweisen Fehlerschranken in 'err' die insgesamt eingehaltene Fehlerschranke zurück.
    Die Fehlerschranke bezieht sich auf die Kürzung hierarchischer Tuckertensoren.
    """
    err[1] = 0
    return np.linalg.norm(np.array(list(err.values())))
