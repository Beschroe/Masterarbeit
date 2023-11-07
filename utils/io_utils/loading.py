import os
import numpy as np
import json
from .mixed import get_endings_of_all_files_in_dir
import pickle


def load(name):
    """
    Laedt den Inhalt von 'name'. Dabei kann es sich entweder um einen np.ndarray,
    ein dict oder einen Ordner handeln. Handelt es sich um einen Ordner, wird davon
    ausgegangen, dass dieser dicts oder np.ndarrays enthaelt.
    """
    if name.endswith(".npy"):
        # np.ndarray Fall
        return load_single_array(name)
    elif name.endswith(".txt"):
        return load_single_dict(name)
    elif name.endswith(".pkl"):
        return load_single_htucker(name)
    elif os.path.isdir(name):
        # Ordner Fall
        # Unterscheide zwischen "dict Ordner", "np.ndarray Ordner" und "htucker Ordner"
        endings = get_endings_of_all_files_in_dir(name)
        if {".txt"} == set(endings):
            # Ordner mit dicts Fall
            return load_dir_of_dicts(name)
        elif {".npy"} == set(endings):
            # Ordner mit np.ndarrays Fall
            return load_dir_of_arrays(name)
        elif {".pkl"} == set(endings):
            return load_dir_of_htucker(name)
        else:
            raise ValueError("'name' ist kein passendes Verzeichnis.")

    else:
        raise ValueError("'name' ist kein gueltiger Datei- bzw. Verzeichnisname.")


def load_dir_of_dicts(dname):
    """
    Erwartet ein Verzeichnis mit .txt Dateien. Diese werden in eine Liste eingelesen und diese Liste wird anschließend
    zurückgegeben. Bezogen auf ihre Namen werden die Dateien in aufsteigender alphabetischer Ordnung und der Länge nach
    sortiert.
    Beispiel: t0.npy, t1.npy, t2.npy, ..., t10.npy, t11.npy, ..., t199.npy
    """
    if os.path.isdir(dname):

        # Pruefe, ob dname Ordner nur txt Dateien enthaelt
        endings = get_endings_of_all_files_in_dir(dname)
        if {".txt"} != set(endings):
            raise ValueError("'dname' ist kein gueltiger Ordner, da er nicht nur .txt Dateien enthaelt.")

        names = sorted(os.listdir(dname))
        names = sorted(names, key=len, reverse=False)

        data = []

        for name in names:
            fname = os.path.join(dname, name)
            data += [load_single_dict(fname)]

        return data

    else:
        raise ValueError("'dname' ist kein gueltiger Ordnername.")


def load_single_dict(fname):
    """
    Gibt das dict zurück, das in der Datei mit Namen 'fname' abgespeichert ist.
    """
    if fname.endswith(".txt"):

        with open(fname, 'r') as f:
            d = json.loads(f.readline())
            return d

    else:
        raise ValueError("'fname' muss auf '.txt' enden.")


def load_single_array(fname):
    """
    Gibt den np.ndarray zurück, der in der Datei mit Namen 'fname' abgespeichert ist.
    """
    # Pruefe fname
    if not fname.endswith(".npy"):
        raise ValueError("'name' muss mit '.npy' enden.")

    with open(fname, "rb") as f:
        arr = np.load(f)

    return arr


def load_dir_of_arrays(dname):
    """
    Erwartet ein Verzeichnis mit .npy Dateien. Diese werden in eine Liste eingelesen und diese Liste wird anschliessend
    zurückgegeben. Bezogen auf ihre Namen werden die Dateien in aufsteigender alphabetischer Ordnung und der Länge nach
    sortiert.
    Beispiel: t0.npy, t1.npy, t2.npy, ..., t10.npy, t11.npy, ..., t199.npy
    """
    if not os.path.isdir(dname):
        raise ValueError("'dname' ist kein gueltiger Ordnername.")
    # Pruefe, ob dname Ordner nur .npy Dateien enthaelt
    endings = get_endings_of_all_files_in_dir(dname)
    if {".npy"} != set(endings):
        raise ValueError("'dname' ist kein gueltiger Ordner, da er nicht nur .txt Dateien enthaelt.")
    names = sorted(os.listdir(dname))
    names = sorted(names, key=len, reverse=False)
    data = []
    for name in names:
        fname = os.path.join(dname, name)
        data += [load_single_array(fname)]
    return data


def load_single_htucker(fname):
    """
    Gibt das Pickle Objekt, dass unter 'fname' abgespeichert ist, zurück
    """
    if not fname.endswith(".pkl"):
        raise ValueError("'fname' muss auf '.pkl' enden.")
    with open(fname, "rb") as f:
        htucker = pickle.load(f)
    return htucker

def load_dir_of_htucker(dname):
    """
    Erwartet ein Verzeichnis mit ausschließlich hierarchischen Tuckertensoren, die als .pkl Dateien vorliegen.
    Diese werden in eine Liste eingelesen und diese Liste wird anschliessend zurückgegeben.
    Bezogen auf ihre Namen werden die Dateien in aufsteigender alphabetischer
    Ordnung und der Laenge nach sortiert.
    Beispiel: t0.pkl, t1.pkl, ..., t10.pkl, t11.pkl, ..., t199.pkl
    """
    if not os.path.isdir(dname):
        raise ValueError("'dname' ist kein gueltiger Verzeichnisname.")
    # Pruefe, ob dname Ordner nur .pkl Dateien enthaelt
    endings = get_endings_of_all_files_in_dir(dname)
    if {".pkl"} != set(endings):
        raise ValueError("'dname' ist kein passendes Verzeichnis, da es nicht nur .pkl Dateien enthält.")
    names = sorted(os.listdir(dname))
    names = sorted(names, key=len, reverse=False)
    data = []
    for name in names:
        fname = os.path.join(dname, name)
        data += [load_single_htucker(fname)]
    return data

