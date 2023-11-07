import numpy as np
from scipy.optimize import newton


def get_s_inf(R0, S0, sigma):
    """
    Gibt den von der Theorie des klassischen SIR-Modells vorhergesagten Grenzwert der Anzahl an Suszeptiblen zurück.
    """
    assert sigma >= 1
    root = None
    for x0 in np.linspace(1e-3, 1, 100):
        buf = -1
        try:
            buf = newton(lambda x: get_s_inf_equation(R0=R0, S0=S0, sigma=sigma, S_inf=x), x0)
        except:
            pass
        if 0 < buf < 1:
            root = buf
            break
    if root is None:
        raise ValueError("S_inf konnte nicht bestimmt werden.")
    return root


def get_s_inf_equation(R0, S0, sigma, S_inf):
    return 1 - R0 - S_inf + (np.log(S_inf / S0) / sigma)
