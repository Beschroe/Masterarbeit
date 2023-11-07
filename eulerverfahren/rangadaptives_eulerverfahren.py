from tqdm import tqdm
from tensor.htucker import HTucker
from utils.misc_utils.get_error import get_error


class RangadaptivesEulerverfahren:
    """
    Rangadaptives Eulerverfahren
    """
    def __init__(self, model, output_handler):
        self.model = model
        self.output_handler = output_handler

    def compute(self):
        self.model.A = self.model.A0
        tau = self.model.t_disc[1] - self.model.t_disc[0]    # Konstante Zeitschrittweite
        for t in tqdm(self.model.t_disc[:-1], smoothing=0):
            # Aktuelle Lösung
            # Berechnung rhs
            rhs_S, rhs_I = self.model.rhs(t)
            # Update der Lösung
            S = HTucker.add(x=self.model.A[0], y=HTucker.scalar_mul(x=rhs_S, a=tau))
            I = HTucker.add(x=self.model.A[1], y=HTucker.scalar_mul(x=rhs_I, a=tau))
            # Kürzen der Lösung
            S, err_S, _ = HTucker.truncate_htucker(x=S, max_rank=self.model.max_rank_r, abs_err=self.model.eps_k)
            I, err_I, _ = HTucker.truncate_htucker(x=I, max_rank=self.model.max_rank_r, abs_err=self.model.eps_k)
            A_next = [S,I]
            # Speichern eines Snapshots für jeden vollen Tag
            if round(t) == t:
                # Uebergabe an den OutputHandler
                self.output_handler.write_snapshot(round(t))
            # Übernahme neuer Lösung
            self.model.A = A_next
            # Update Error
            self.model.error["S"] = get_error(err_S)
            self.model.error["I"] = get_error(err_I)
            # Update rank
            self.model.rank["S"] = max(S.rank.values())
            self.model.rank["I"] = max(I.rank.values())

        # Schreibe letzte Lösung
        self.output_handler.write_solution(round(self.model.t_disc[-1]))

