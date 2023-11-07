import numpy as np
from pandas import read_csv
import pathlib
from os.path import join

CONTACT_MATRIX = join(pathlib.Path(__file__).parent.resolve(), "Kontaktmatrix.csv")
KOMPATIBILITAET_MATRIX = np.array([[1, 1, 1, 1],
                                   [0, 1, 0, 1],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1]])


def get_prob_matrix(beta_avg, delta, f_B):
    """
    Berechnet die Infektionswahrscheinlichkeit eines Kontaktes unter Berücksichtigung
    einer Durchschnittswahrscheinlichkeit 'beta_avg' und den Blutgruppen der beteiligten Individuen.
    Hintergrund: Es wird die Annahme getroffen, dass bei inkompatiblen Blutgruppen der beteiligten
    Individuen eines Kontaktes, die Übertragungswahrscheinlichkeit um delta*100 % niedriger ausfällt.
        - 'beta' ist ein np.ndarray mit shape (4,4) -> (AB, A, B, 0) x (AB, A, B, 0).
        - Die 4 Zeilen stehen fuer die Blutgruppe des suszeptiblen Individuums.
        - Die 4 Spalten stehen fuer die Blutgruppe des infektioesen Individuums.
        - Entsprechend enthaelt beta[1,2] die Infektionswahrscheinlichkeit
          bei einem Kontakt zwischen einem suszeptiblen Individuum mit Blutgruppe A
          und einem infektiösen Individuum mit Blutgruppe B.
    """
    f_BB = np.tensordot(f_B, f_B, axes=0)
    inkompatibilitaet_matrix = np.ones((4, 4)) - KOMPATIBILITAET_MATRIX
    beta_kompatibel = beta_avg / np.tensordot(f_BB, (1 - delta) * inkompatibilitaet_matrix + KOMPATIBILITAET_MATRIX)
    beta_inkompatibel = (1 - delta) * beta_kompatibel
    beta = beta_kompatibel * KOMPATIBILITAET_MATRIX + beta_inkompatibel * inkompatibilitaet_matrix
    if not np.all(beta <= 1):
        raise ValueError("'beta_avg' und 'delta' passen nicht zusammen, da so Wahrscheinlichkeiten > 1 entstehen.")
    return beta


def get_contact_matrix(a=1):
    """
    Berechnet die Standardkontaktmatrix auf Grundlage der Datei CONTACT_MATRIX (siehe Konstanten).
    kappa[a1,a2] ist durchschnittliche Anzahl an Kontakten eines Individuums aus Altersklasse a2 mit Individuen
    aus Altersklasse a1.
    """
    kappa = a * read_csv(CONTACT_MATRIX, index_col=0).to_numpy()
    return kappa.T


def get_lamda(kappa, beta, f_B, f_NAB):
    """
    Berechnet die Kontaktrate für das erweiterte SIR-Modell.
    Berechnet ferner die skalierte Kontaktrate für das erweiterte SIR-Modell gemäß.
    """
    lamda = np.tensordot(kappa, beta, axes=0)
    lamda = np.swapaxes(lamda, 1, 2)
    lamda = lamda * f_B[None, :, None, None]
    lamda_scaled = lamda * (1 / (f_NAB[:, :, 0, 0])[:, :, None, None])
    return lamda, lamda_scaled


def get_lamda_scaled(kappa, beta, f_B, f_AB, N):
    """
    Berechnet die skalierte Kontaktrate für das erweiterte SIR-Modell ohne Diffusion.
    """
    lamda = np.tensordot(kappa, beta, axes=0)
    lamda = np.swapaxes(lamda, 1, 2)
    lamda = lamda * f_B[None, :, None, None]
    lamda = lamda / (N * f_AB)[:, :, None, None]
    return lamda


def get_lamda_avg(lamda, f_AB):
    """
    Berechnet die durchschnittliche Kontaktrate. f_AB enthält die Anteile an Individuen mit allen Kombinationen
    aus Altersklasse und Blutgruppe.
    lamda_avg := Σ_a' Σ_b' (Σ_a Σ_b lamda[a,b,a',b']) * f_AB[a',b']
    """
    lamda_avg = np.tensordot(f_AB, np.sum(lamda, axis=(0, 1)))
    return lamda_avg


def get_sigma_avg(lamda, gamma, f_AB):
    """
    Berechnet die durchschnittliche Kontaktzahl sigma.
    sigma :=  Σ_a' Σ_b' (Σ_a Σ_b lamda[a,b,a',b']/gamma[a']) * f_AB[a',b']
    """
    sigma = np.sum(lamda * (1 / gamma)[None, None, :, None], axis=(0, 1))
    sigma_avg = np.tensordot(f_AB, sigma)
    return sigma_avg


def get_i_max(sigma, R0, S0):
    """
    Berechnet die maximale Prävalenz für gegebene Anfangswerte 'R0', 'S0' und die Kontaktzahl 'sigma'.
    """
    return 1 - R0 - 1 / sigma - np.log(sigma * S0) / sigma
