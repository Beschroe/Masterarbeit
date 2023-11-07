import numpy as np


def full(self):
    """
    Rekonstruiert aus dem gegebenen hierarchischen Tuckertensor den expliziten vollen Tensor.
    """
    rebuilt_as_vector = self.rebuild_tensor_helper(0)
    root_dim = self.dtree.get_dim(0)
    root_shape = np.array(self.shape)[root_dim]
    reshaped_to_tensor = rebuilt_as_vector.reshape(root_shape, order="F")
    moved_axes_tensor = np.moveaxis(reshaped_to_tensor, source=range(self.order), destination=root_dim)
    return moved_axes_tensor


def rebuild_tensor_helper(self, node):
    """
    Helferfunktion
    """
    dt = self.dtree
    if dt.is_leaf(node):
        # Basisfall
        # Node ist ein Blatt
        return self.U[node]
    # Rekursiver Fall
    right = dt.get_right(node)
    left = dt.get_left(node)
    Ul = self.rebuild_tensor_helper(left)
    Ur = self.rebuild_tensor_helper(right)
    B = self.B[node]
    prod = np.tensordot(Ul, B, axes=[1, 0])
    prod = np.tensordot(Ur, prod, axes=[1, 1])
    return prod.reshape((-1, B.shape[2]))
