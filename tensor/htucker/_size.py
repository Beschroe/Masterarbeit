def get_size(self):
    """
    Gibt den Speicherplatzbedarf der Blattmatrizen und Transfertensoren in Bytes zurück.
    """
    U = self.U
    B = self.B
    size = 0
    for u in U.values():
        size += u.nbytes
    for b in B.values():
        size += b.nbytes
    return size
