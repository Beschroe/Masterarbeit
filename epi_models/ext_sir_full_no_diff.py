import numpy as np


class ExtendedSIRModelNoDiff:
    """
    Beschränktes erweitertes SIR-Modell basierend auf vollen Tensoren. Bei ausgeschalteter Diffusionskomponente
    wird nur der Reaktionsanteil in einem Raumpunkt berechnet.
    """
    def __init__(self, A0, lamda, gamma):
        self.A0 = A0
        self.lamda = lamda
        self.gamma = gamma
        self.A = None
        self.t_disc = None
        self.N = None
        self.f_A = None
        self.f_B = None
        self.f_AB = None

    def set_population_settings(self, N, f_A, f_B, f_AB):
        self.N = N
        self.f_A, self.f_B = f_A, f_B
        self.f_AB = f_AB

    def set_time_discretization(self, T, Nt):
        self.t_disc = np.arange(0, T + T / Nt, T / Nt)

    def reaction(self):
        S, I = self.A
        S2I = S * np.tensordot(self.lamda, I, axes=((2, 3), (0, 1)))
        I2R = self.gamma[:, None] * I
        reaction = np.stack((-S2I, S2I - I2R))
        return reaction

    def rhs(self, t):
        return self.reaction()
