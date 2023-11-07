import numpy as np
import os


def int_keys_to_str(d):
    """
    Bekommt ein (eventuell nested) dict und konvertiert alle int Schluessel zu str.
    """
    d_new = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = int_keys_to_str(value)
        if np.issubdtype(type(key), int):
            key = str(key)
        d_new[key] = value
    return d_new


def get_endings_of_all_files_in_dir(dname):
    """
    Gibt eine Liste mit allen Dateiendungen aller Dateien eines Ordners zurueck.
    'dname' ist der Name dieses Ordners.
    """
    if os.path.isdir(dname):
        endings = []
        for fname in os.listdir(dname):
            fn = os.path.join(dname, fname)
            if os.path.isfile(fn):
                filename, extension = os.path.splitext(fn)
                endings += [extension]
            else:
                raise ValueError("'dname' ist kein gueltiger Ordner, da dieser nicht nur Dateien enthaelt.")
    else:
        raise ValueError("'dname' ist kein gueltiger Ordnername.")
    return endings
