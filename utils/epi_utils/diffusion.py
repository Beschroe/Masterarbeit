import numpy as np


def neumann_diff_matrix(n):
    """
    Gibt die Matrix zurück, mit der der diskrete Laplace-Operator unter Einhaltung der Neumann Randbedingung berechnet
    werden kann zurück.
    """
    A = []

    # Erste Zeile von A
    first_row = np.zeros(n)
    first_row[0] = 1
    first_row[1] = -2
    first_row[2] = 1
    A += [first_row]

    # Innere Zeilen von A
    for i in range(1, n - 1):
        row = np.zeros(n)
        row[i - 1] = 1
        row[i] = -2
        row[i + 1] = 1
        A += [row]

    # Letzte Zeile von A
    last_row = np.zeros(n)
    last_row[-1] = -(7 / 2)
    last_row[-2] = 4
    last_row[-3] = -(1 / 2)
    A += [last_row]

    return np.array(A)
