import numpy as np
from utils.epi_utils.diffusion import neumann_diff_matrix
from tensor.htucker import HTucker
from utils.misc_utils.get_error import get_error


class ExtendedSIRModelHTucker:
    """
    Erweitertes SIR-Modell auf Basis hierarchischer Tuckertensoren.
    """
    def __init__(self, A0, lamda, gamma, D):
        self.A0 = A0
        self.lamda = lamda
        self.gamma = gamma
        self.D = D
        self.rank = {"S": 0, "I": 0}
        self.error = {"S": 0, "I": 0}
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
        self.eps_k = None
        self.eps_r = None
        self.max_rank_k = None
        self.max_rank_r = None

    def set_population_settings(self, N, f_N, f_A, f_B, f_AB, f_NAB):
        self.N, self.f_N = N, f_N
        self.f_A, self.f_B = f_A, f_B
        self.f_AB, self.f_NAB = f_AB, f_NAB

    def set_htucker_settings(self, mrs, mrd, aes, aed):
        self.max_rank_k, self.max_rank_r = mrs, mrd
        self.eps_k, self.eps_r = aes, aed

    def truncate_A0(self, enable_truncation_info=False):
        S0, err_bnd_S0, _ = HTucker.truncate(A=self.A0[0], max_rank=self.max_rank_k, abs_err=self.eps_k)
        I0, err_bnd_I0, _ = HTucker.truncate(A=self.A0[1], max_rank=self.max_rank_k, abs_err=self.eps_k)
        if enable_truncation_info:
            print("Fehlerschranke S0: ", get_error(err_bnd_S0))
            print("Fehlerschranke I0: ", get_error(err_bnd_I0))
        self.A0 = [S0, I0]
        # Update error
        self.error["S"] = get_error(err_bnd_S0)
        self.error["I"] = get_error(err_bnd_I0)
        # Update rank
        self.rank["S"] = max(S0.rank.values())
        self.rank["I"] = max(I0.rank.values())

    def truncate_lamda(self, enable_truncation_info=False):
        lamda_ht, err_bnd_lamda, _ = HTucker.truncate(A=self.lamda, max_rank=80, abs_err=1e-6)
        if enable_truncation_info:
            print("Fehlerschranke lamda: ", get_error(err_bnd_lamda))
        self.lamda = lamda_ht

    def set_time_discretization(self, T, Nt):
        self.t_disc = np.arange(0, T + T/Nt, T / Nt)
        if self.h is not None and self.D is not None:
            # check CFL
            tau = T / Nt
            assert tau < self.h ** 2 / (self.D * 2)

    def set_space_discretization(self, X, Y, Nx, Ny):
        self.X, self.Y = X, Y
        self.Nx, self.Ny = Nx, Ny
        assert self.X / (self.Nx - 1) == self.Y / (self.Ny - 1)
        self.h = self.X / (self.Nx - 1)
        if self.t_disc is not None and self.D is not None:
            # check CFL
            Nt, T = self.t_disc.shape[0], self.t_disc[-1]
            tau = T / Nt
            assert tau * self.D / self.h ** 2 <= 1 / 2
        self.diffx = self.D / (self.h ** 2) * neumann_diff_matrix(self.Nx)
        self.diffy = self.D / (self.h ** 2) * neumann_diff_matrix(self.Nx)

    def reaction(self):
        S, I = self.A
        lamdaI = HTucker.contract(x=self.lamda, y=I, dims_x=[2, 3], dims_y=[0, 1])
        lamdaI, err_bnd_lamdaI, _ = HTucker.truncate_htucker(x=lamdaI, max_rank=self.max_rank_r,
                                                             abs_err=self.eps_r)
        mr = max(lamdaI.rank.values()) * max(S.rank.values())
        S2I, _, _ = HTucker.ews_multiplication(x=S, y=lamdaI, max_rank=mr, abs_err=self.eps_k)
        S2I, _, _ = HTucker.truncate_htucker(x=S2I, max_rank=self.max_rank_r, abs_err=self.eps_r)
        I2R = HTucker.ews_mode_multiplication(x=I, vec=self.gamma, mu=0)
        return S2I, I2R

    def diffusion(self):
        S, I = self.A
        diff_Sx = HTucker.mode_multiplication(S, self.diffx, 2)
        diff_Ix = HTucker.mode_multiplication(I, self.diffx, 2)
        diff_Sy = HTucker.mode_multiplication(S, self.diffy, 3)
        diff_Iy = HTucker.mode_multiplication(I, self.diffy, 3)
        return diff_Sx, diff_Ix, diff_Sy, diff_Iy

    def rhs(self, t):
        S2I, I2R = self.reaction()
        diff_Sx, diff_Ix, diff_Sy, diff_Iy = self.diffusion()
        rhs_S, _, _ = HTucker.add_and_truncate(summanden=[HTucker.scalar_mul(x=S2I, a=-1), diff_Sx, diff_Sy],
                                                   max_rank=self.max_rank_r, abs_err=self.eps_r, copy=False)
        rhs_I, _, _ = HTucker.add_and_truncate(summanden=[S2I, HTucker.scalar_mul(x=I2R, a=-1), diff_Ix, diff_Iy],
                                                   max_rank=self.max_rank_r, abs_err=self.eps_r, copy=False)
        return [rhs_S, rhs_I]
