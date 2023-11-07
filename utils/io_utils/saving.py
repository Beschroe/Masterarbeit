from .mixed import int_keys_to_str
import os
import numpy as np
import json
import pickle
from tensor.htucker import HTucker


def save(fname, data):
    """
    Speichert das Datum 'data' in einer Datei mit dem Namen 'fname' ab.
    Es wird erwartet, dass 'data' entweder vom Typ np.ndarray, dict oder float ist.
    """
    if isinstance(data, np.ndarray):
        save_array(fname, data)
    elif isinstance(data, dict):
        save_dict(fname, data)
    elif isinstance(data, HTucker):
        save_object(fname, data)
    elif np.issubdtype(type(data), float):
        save_array(fname, np.array(data))
    elif np.issubdtype(type(data), int):
        save_array(fname, np.array(data))
    else:
        raise ValueError("'data' muss entweder ein np.ndarray, HTucker Objekt, dict, float oder int sein.")
    return


def save_array(fname, arr):
    """
    Speichert den np.ndarray 'arr' in das File fname vom Typ .npy.
    Falls dieses File bereits np.arrays enthaelt, wird 'arr' angehaengt.
    """
    # Pruefe fname
    if not fname.endswith(".npy"):
        raise ValueError("'fname' muss mit '.npy' enden.")
    # Pruefe arr
    if not isinstance(arr, np.ndarray) and not np.issubdtype(type(arr), float)\
            and not np.issubdtype(type(arr), int):
        raise ValueError("'arr' muss entweder ein np.ndarray, float oder int sein.")

    if np.issubdtype(type(arr), float):
        arr = np.array(arr)
    if np.issubdtype(type(arr), int):
        arr = np.array(arr)

    try:
        with open(fname, "wb") as f:
            np.save(f, arr)
    except Exception:
        os.remove(fname)
        raise
    return


def save_dict(fname, d):
    # Pruefe fname
    if not fname.endswith(".txt"):
        raise ValueError("'fname' muss auf '.txt' enden.")
    if not isinstance(d, dict):
        raise ValueError("'d' muss vom Typ dict sein.")
    d = int_keys_to_str(d)
    try:
        with open(fname, 'w') as f:
            f.write(f"{json.dumps(d)}\n")
    except Exception:
        os.remove(fname)
        raise
    return

def save_object(fname, obj):
    """
    Speichere das Objekt unter dem Pfad 'fname' als .pkl Datei unter Nutzung des Pickle Moduls der Python
    standard library.
    """
    if not fname.endswith(".pkl"):
        raise ValueError("'fname' muss auf '.pkl' enden.")
    try:
        with open(fname, 'wb') as f:
            pickle.dump(obj, f, -1)
    except Exception:
        os.remove(fname)
        raise
    return
