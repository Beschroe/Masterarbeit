import numpy as np
import types
import os
from .saving import save
from tensor.htucker import HTucker


class OutputHandler:

    def __init__(self, path, dname, model, standard_output=True):
        if not isinstance(path, str):
            raise TypeError("'path' vom Typ str sein.")
        if not isinstance(dname, str):
            raise TypeError("'dname' vom Typ str sein.")
        self.path = path
        self.dname = dname
        self.output_funcs_sol = []
        self.output_funcs_else = []
        self.subdnames = {}
        self.model = model
        self.htucker = self.is_htucker()
        self.shape = self.get_shape()
        self.helper_htucker_spatial = None
        self.helper_htucker_total = None
        if standard_output:
            self.set_standard_output_funcs()

    def get_shape(self):
        """
        Gibt die Größen der Modi des Modells zurück: (Na,Nb,Nx,Ny)
        """
        return self.model.A0[0].shape

    def is_htucker(self):
        """
        Gibt zurück, ob das Modell mit vollen oder hierarchischen Tuckertensoren arbeitet.
        """
        return isinstance(self.model.A0[0], HTucker)

    def set_output_func(self, output_func, dname, is_solution=True):
        if not isinstance(output_func, types.FunctionType):
            raise TypeError("'output_func' muss eine Funktion sein.")
        if not isinstance(dname, str):
            raise TypeError("'dname' muss vom Typ str sein.")
        if is_solution:
            self.output_funcs_sol += [output_func]
        else:
            self.output_funcs_else += [output_func]
        self.subdnames[output_func] = dname

    def prepare_dir(self):
        # Erstelle Wurzelverzeichnis
        os.makedirs(os.path.join(self.path, self.dname), exist_ok=False)
        # Erstelle Unterverzeichnisse für die Ergebnisse
        for func in self.output_funcs_sol:
            # Entferne endung
            subdname = self.subdnames[func].rsplit(".", 1)[0]
            os.makedirs(os.path.join(self.path, self.dname, subdname), exist_ok=False)
        for func in self.output_funcs_else:
            subdname = self.subdnames[func].rsplit(".", 1)[0]
            os.makedirs(os.path.join(self.path, self.dname, subdname), exist_ok=False)

    def set_standard_output_funcs(self):
        if self.htucker:
            self.set_standard_output_funcs_htucker()
        else:
            self.set_standard_output_funcs_full()

    def set_standard_output_funcs_full(self):
        # Lösung
        self.set_output_func(lambda model: model.A[0], "S.npy", is_solution=True)
        self.set_output_func(lambda model: model.A[1], "I.npy", is_solution=True)

    def set_standard_output_funcs_htucker(self):
        # Lösung
        self.set_output_func(lambda model: model.A[0], "S.pkl", is_solution=True)
        self.set_output_func(lambda model: model.A[1], "I.pkl", is_solution=True)
        # Error
        self.set_output_func(lambda model: model.error["S"], "Error_S.npy", is_solution=True)
        self.set_output_func(lambda model: model.error["I"], "Error_I.npy", is_solution=True)

    def write_settings(self, instance_variables):
        if not isinstance(instance_variables, dict):
            raise TypeError("'instance_variables' muss ein dict sein.")
        iv = instance_variables
        to_write = {}
        # Speichern von lambda, gamma, D, T, tau, X, Y, h, N, f_N, f_A und f_B
        to_save = ["lamda", "gamma", "X", "D", "Y", "h", "N", "f_N", "f_A", "f_B"]
        for k in to_save:
            if k in iv.keys():
                v = iv[k]
                if isinstance(v, np.ndarray):
                    to_write[k] = v.tolist()
                elif isinstance(v, HTucker):
                    to_write[k] = v.full().tolist()
                else:
                    to_write[k] = v
        # Separates speichern von T, tau
        T = iv["t_disc"][-1]
        tau = iv["t_disc"][1] - iv["t_disc"][0]
        to_write["T"] = T
        to_write["tau"] = tau
        fname_complete = os.path.join(self.path, self.dname, "settings.txt")
        save(fname_complete, to_write)

    def write_snapshot(self, t):
        """
        Schreibt einen Schnappschuss für den aktuellen Zeitpunkt. Dieser beinhaltet neben der Lösung zum Zeitpunkt
        't' weitere Informationen wie beispielsweise den Speicherbedarf.
        """
        for func in self.output_funcs_sol:
            fname = self.subdnames[func].rsplit(".", 1)[0] + "_t" + str(t) + "." + self.subdnames[func].rsplit(".", 1)[1]
            fname_complete = os.path.join(self.path, self.dname, self.subdnames[func].rsplit(".", 1)[0], fname)
            save(fname_complete, func(self.model))
        for func in self.output_funcs_else:
            fname = self.subdnames[func].rsplit(".", 1)[0] + "_t" + str(t) + "." + self.subdnames[func].rsplit(".", 1)[1]
            fname_complete = os.path.join(self.path, self.dname, self.subdnames[func].rsplit(".", 1)[0], fname)
            save(fname_complete, func(self.model))

    def write_solution(self, t):
        """
        Schreibt die Lösung zum Zeitpunkt 't'.
        """
        for func in self.output_funcs_sol:
            fname = self.subdnames[func].rsplit(".", 1)[0] + "_t" + str(t) + "." + self.subdnames[func].rsplit(".", 1)[1]
            fname_complete = os.path.join(self.path, self.dname, self.subdnames[func].rsplit(".", 1)[0], fname)
            save(fname_complete, func(self.model))