import numpy as np
from utils.epi_utils.diffusion import neumann_diff_matrix


class ExtendedSIRModelFull:
    """
    Erweitertes SIR-Modell auf Basis voller Tensoren.
    """

    def __init__(self, A0, lamda, gamma, D):
        self.A0 = A0
        self.lamda = lamda
        self.gamma = gamma
        self.D = D
        self.A = None
        self.t_disc = None
        self.X = None
        self.Y = None
        self.Nx = None
        self.Ny = None
        self.diffx = None
        self.diffy = None
        self.h = None
        self.N = None
        self.f_N = None
        self.f_A = None
        self.f_B = None
        self.f_AB = None
        self.f_NAB = None

    def set_population_settings(self, N, f_N, f_A, f_B, f_AB, f_NAB):
        self.N, self.f_N = N, f_N
        self.f_A, self.f_B = f_A, f_B
        self.f_AB, self.f_NAB = f_AB, f_NAB

    def set_time_discretization(self, T, Nt):
        self.t_disc = np.arange(0, T +T/Nt, T / Nt)
        if self.h is not None and self.D is not None:
            # check CFL
            tau = T / Nt
            assert tau < self.h ** 2 / (self.D * 2)

    def set_space_discretization(self, X, Y, Nx, Ny, diffusion=True):
        self.X, self.Y = X, Y
        self.Nx, self.Ny = Nx, Ny
        assert self.X / (self.Nx-1) == self.Y / (self.Ny-1)
        if diffusion:
            self.h = self.X / (self.Nx-1)
            if self.t_disc is not None and self.D is not None:
                # check CFL
                Nt, T = self.t_disc.shape[0], self.t_disc[-1]
                tau = T / Nt
                assert tau * self.D / self.h ** 2 <= 1 / 2
            self.diffx = self.D / (self.h ** 2) * neumann_diff_matrix(self.Nx)
            self.diffy = self.D / (self.h ** 2) * neumann_diff_matrix(self.Nx)
        else:
            self.h = 0

    def reaction(self):
        S, I = self.A
        S2I = S * np.tensordot(self.lamda, I, axes=((2, 3), (0, 1)))
        I2R = self.gamma[:, None, None, None] * I
        reaction = np.stack((-S2I, S2I - I2R))
        return reaction

    def diffusion(self):
        diffusion_Ax = np.tensordot(self.A, self.diffx, axes=[3, 1])
        diffusion_Ax = np.moveaxis(diffusion_Ax, source=[4], destination=[3])
        diffusion_Ay = np.tensordot(self.A, self.diffy, axes=[4, 1])
        diffusion_A = diffusion_Ax + diffusion_Ay
        return diffusion_A

    def rhs(self, t):
        return self.reaction() + self.diffusion()
