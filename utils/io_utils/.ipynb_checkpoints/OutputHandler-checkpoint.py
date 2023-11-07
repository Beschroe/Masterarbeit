import numpy as np
import types
from Utils.GetSize import get_size
import os
from .saving import save
import shutil


class OutputHandler:

    def __init__(self, path, dname, standard_output=True):
        if not isinstance(path, str):
            raise TypeError("'path' vom Typ str sein.")
        if not isinstance(dname, str):
            raise TypeError("'dname' vom Typ str sein.")
        self.path = path
        self.dname = dname
        self.output_funcs = []
        self.subdnames = {}
        if standard_output:
            self.set_standard_output_funcs()

    def set_output_func(self, output_func, dname):
        if not isinstance(output_func, types.FunctionType):
            raise TypeError("'output_func' muss eine Funktion sein.")
        if not isinstance(dname, str):
            raise TypeError("'dname' muss vom Typ str sein.")
        self.output_funcs += [output_func]
        self.subdnames[output_func] = dname

    def prepare_dir(self, remove_if_exists=False):
        # Erstelle Wurzelverzeichnis
        try:
            os.makedirs(os.path.join(self.path, self.dname), exist_ok=False)
        except FileExistsError e:
            if remove_if_exists:
                shutil.rmtree(os.path.join(self.path, self.dname))
                self.prepare_dir(False)
            else:
                raise e
        # Erstelle Unterverzeichnisse
        for func in self.output_funcs:
            subdname = self.subdnames[func]
            os.makedirs(os.path.join(self.path, self.dname, subdname), exist_ok=False)

    def set_standard_output_funcs(self):
        # Spatial
        self.set_output_func(lambda data: np.sum(data[0][0], axis=(0, 1)), "S_spatial")
        self.set_output_func(lambda data: np.sum(data[0][1], axis=(0, 1)), "I_spatial")
        # Total
        self.set_output_func(lambda data: np.sum(data[0][0]), "S_total")
        self.set_output_func(lambda data: np.sum(data[0][1]), "I_total")
        # Memory
        self.set_output_func(lambda data: get_size(data[0][0] / 1e6), "Memory_S")
        self.set_output_func(lambda data: get_size(data[0][1] / 1e6), "Memory_I")
        # Memory Peak
        self.set_output_func(lambda data: data[1] / 1e6, "Memory_Peak")

    def write_settings(self, instance_variables):
        if not isinstance(instance_variables, dict):
            raise TypeError("'instance_variables' muss ein dict sein.")
        iv = instance_variables
        disc = {"tau": iv["tau"], "Nt": iv["Nt"], "h": iv["h"], "Nx": iv["Nx"], "Ny": iv["Ny"]}
        parameter = {"lamda": iv["lamda"].full().tolist(), "gamma": iv["gamma"].tolist()}
        settings = {"disc": disc, "parameter": parameter}
        fname_complete = os.path.join(self.path, self.dname, "settings.txt")
        save(fname_complete, settings)

    def write_snapshot(self, t, data):
        # t: Zeitpunkt
        # data: [[S,I], Memory_peak]
        for func in self.output_funcs:
            fname = self.subdnames[func] + "_t" + str(t) + ".npy"
            fname_complete = os.path.join(self.path, self.dname, self.subdnames[func], fname)
            save(fname_complete, func(data))
