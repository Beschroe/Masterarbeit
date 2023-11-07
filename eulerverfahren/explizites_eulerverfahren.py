from tqdm import tqdm


class ExplizitesEulerverfahren:
    """
    Explizites Eulerverfahren
    """
    def __init__(self, model, output_handler):
        self.model = model
        self.output_handler = output_handler

    def compute(self):
        self.model.A = self.model.A0
        tau = self.model.t_disc[1] - self.model.t_disc[0]   # Konstante Zeitschrittweite
        for t in tqdm(self.model.t_disc[:-1], smoothing=0):
            A_new = self.model.A + tau * self.model.rhs(t)
            # Speichern eines Snapshots für jeden vollen Tag
            if round(t) == t:
                # Übergabe an den OutputHandler
                self.output_handler.write_snapshot(round(t))
            self.model.A = A_new
        # Schreibe letzte Lösung
        self.output_handler.write_solution(round(self.model.t_disc[-1]))

