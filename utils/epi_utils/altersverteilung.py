from pandas import read_csv
import pathlib
import numpy as np
from os.path import join

pathlib.Path(__file__).parent.resolve()
FILE_NAME = join(pathlib.Path(__file__).parent.resolve(), "Altersverteilung.csv")


def get_altersverteilung():
    """
    Gibt die Altersklassenverteilung basierend auf dem File FILE_NAME zurück.
    """
    df_altersverteilung = read_csv(FILE_NAME)
    anteile = df_altersverteilung["Anteil"].to_numpy()
    return np.array([round(group.sum(), 5) for group in np.split(anteile, anteile.shape[0] / 5)])
