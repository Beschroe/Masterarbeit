import numpy as np


def trunc_rank(s, max_rank, abs_err=None, rel_err=None):
    """
    Berechnet den minimalen Rang, der die von abs_err und rel_err definierten
    Fehlerschranken erfüllt. Ist dieser größer als max_rank, wird trotzdem max_rank
    zurückgegeben.
    """

    # Check arguments
    if not isinstance(s, np.ndarray):
        raise TypeError("'s' muss ein positiver 1D-np.ndarray mit mindestens einem Eintrag sein.")
    if len(s.shape) != 1:
        raise ValueError("'s' muss ein positiver 1D-np.ndarray mit mindestens einem Eintrag sein.")
    if not all(s.shape[i] > 0 for i in range(len(s.shape))):
        raise ValueError("'s' muss ein positiver 1D-np.ndarray mit mindestens einem Eintrag sein.")
    if not np.issubdtype(type(max_rank), int):
        raise TypeError("'max_rank' muss ein int mit 'max_rank'>= 1 sein.")
    if not max_rank >= 1:
        raise ValueError("'max_rank' muss ein int mit 'max_rank'>= 1 sein.")
    if abs_err is not None:
        if not np.issubdtype(type(abs_err), float):
            raise TypeError("'abs_err' muss ein positiver float sein.")
        if not abs_err > 0:
            raise ValueError("'abs_err' muss ein positiver float sein.")
    if rel_err is not None:
        if not np.issubdtype(type(rel_err), float):
            raise TypeError("'rel_err' muss ein positiver float sein.")
        if not rel_err > 0:
            raise ValueError("'rel_err' muss ein positiver float sein.")

    max_rank = min(max_rank, len(s))

    # Werden die ersten k Singulärvektoren mitgenommen, ist der Fehler in Frobeniusnurm durch s_sum[k] gegeben
    s_sum = np.sqrt(np.cumsum((s ** 2)[::-1]))[::-1]

    # Die Variable sat zeigt an, ob die Fehlerschranke bei gegebenem maximalen Rang eingehalten werden konnte
    sat = True

    # Fallunterscheidungen abhängig davon, welche Constraints durch die Fehlerschranken gesetzt sind
    if rel_err is None:
        if abs_err is None:
            # max_rank wird als fixer Rang zurückgegeben, da weder
            # abs_err noch rel_err constraints definieren
            s_sum = list(s_sum) + [0]
            error = s_sum[max_rank]
            return max_rank, error, True
        else:
            # Nur abs_err bedingt constraint
            k_abs = np.argmax(s_sum < abs_err)
            if k_abs == 0:
                if s_sum[0] >= abs_err:
                    sat = False
                    k_abs = len(s)
                else:
                    k_abs = 1
            k = min(k_abs, max_rank)
            if k < k_abs:
                sat = False
            s_sum = list(s_sum) + [0]
            error = s_sum[k]
            return k, error, sat

    else:
        # rel_err bedingt constraint
        if abs_err is None:
            # Nur rel_err bedingt constraint
            k_rel = np.argmax(s_sum < rel_err * np.linalg.norm(s))
            if k_rel == 0:
                if s_sum[0] >= rel_err * np.linalg.norm(s):
                    sat = False
                    k_rel = len(s)
                else:
                    k_rel = 1
            k = min(k_rel, max_rank)
            if k < k_rel:
                sat = False
            # Get error
            s_sum = list(s_sum) + [0]
            error = s_sum[k]
            return k, error, sat
        else:
            # abs_err und rel_err bedingen constraint
            k_rel = np.argmax(s_sum < rel_err * np.linalg.norm(s))
            if k_rel == 0:
                if s_sum[0] >= rel_err * np.linalg.norm(s):
                    sat = False
                    k_rel = len(s)
                else:
                    k_rel = 1
            k_abs = np.argmax(s_sum < abs_err)
            if k_abs == 0:
                if s_sum[0] >= abs_err:
                    sat = False
                    k_abs = len(s)
                else:
                    k_abs = 1
            k_tol = max(k_rel, k_abs)
            k = min(max_rank, k_tol)
            if k < k_tol:
                sat = False
            s_sum = list(s_sum) + [0]
            error = s_sum[k]
            return k, error, sat
