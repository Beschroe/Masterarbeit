import numpy as np
from tqdm import tqdm


class SIRClassic:
    """
    Klassisches SIR-Modell
    """

    def __init__(self, A0, N, lamda, gamma):
        self.t_disc = None
        self.A0 = A0
        self.N = N
        self.A = None
        self.lamda = lamda
        self.gamma = gamma

    def set_time_discretization(self, T, Nt):
        self.t_disc = np.arange(0, T+T/Nt, T / Nt)

    def get_rhs(self):
        delta_S = - self.A[0] * self.lamda * self.A[1]
        delta_I = self.A[0] * self.lamda * self.A[1] - self.A[1] * self.gamma
        return np.stack((delta_S, delta_I))

    def compute(self):
        self.A = self.A0
        sol = []
        tau = self.t_disc[1] - self.t_disc[0]  # Konstante Zeitschrittweite
        for t in tqdm(self.t_disc[:-1]):
            A_new = self.A + tau * self.get_rhs()
            if round(t) == t:
                sol += [self.A]
            self.A = A_new
        sol += [self.A]
        return np.array(sol)
