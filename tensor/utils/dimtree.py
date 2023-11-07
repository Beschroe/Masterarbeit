import numpy as np
from copy import deepcopy


class dimtree:
    """
    Implementiert den Dimensionsbaum zu einer gegebenen Modusstruktur.
    """

    def __init__(self, children, dim2ind):
        self.children = children
        self.nr_nodes = len(self.children)
        self.dim2ind = dim2ind
        self.nr_dims = len(dim2ind)
        self.leaves = np.sort(np.array(dim2ind))
        self.parent = np.array([-1] * self.nr_nodes)
        for idx, ch in enumerate(self.children):
            if ch[0] == -1:
                continue
            self.parent[ch[0]] = idx
            self.parent[ch[1]] = idx
        self.level = self._construct_level()

    @staticmethod
    def get_canonic_dimtree(ndims):
        """
        Gibt den auf 'ndims' basierenden kanonischen Dimensionsbaum zurück.
        """
        d = ndims
        p = np.ceil(np.log2(d))
        nr_nodes = 2 * d - 1
        children = np.array([[i, i + 1] for i in range(1, nr_nodes, 2)] + [[-1, -1]] * d, dtype=int)
        dim2ind = np.zeros(d, dtype=int)
        for i in range(d):
            if i < 2 * d - 2 ** p:
                dim2ind[i] = 2 ** p - 1 + i
            else:
                dim2ind[i] = 2 ** p - 1 - d + i
        return dimtree(children=children, dim2ind=dim2ind)

    def _construct_level(self):
        level = [0] * self.nr_nodes
        nodes = [(self.children[0][0], 1), (self.children[0][1], 1)]
        while len(nodes) > 0:
            node, lvl = nodes.pop()
            level[node] = lvl
            if not self.is_leaf(node):
                nodes += [(self.children[node][0], lvl + 1), (self.children[node][1], lvl + 1)]
        return np.array(level)

    def print(self):
        """
        Druckt eine Repräsentation des Dimensionsbaums.
        """
        nodes = list(self.get_nodes_dfs(0))[::-1]
        depth = self.get_depth()
        while len(nodes) > 0:
            ind = nodes.pop()
            dim = self.get_dim(ind)
            level = self.get_level(ind)
            print_dim = ""
            if len(dim) > 1:
                print_dim = str(min(dim)) + "-" + str(max(dim))
            else:
                print_dim = str(dim[0]) + " "
            print("\t" * level, print_dim, "\t" * (depth - level + 1), ind)

    def copy(self):
        """
        Gibt eine Kopie des Dimensionsbaums zurück.
        """
        return deepcopy(self)

    def get_dim2ind(self):
        """
        Gibt das Dimension zu Index Mapping zurück.
        """
        return self.dim2ind

    def get_parent(self, ind=-1):
        """
        Gibt den Index des Elternknotens von 'ind' zurück.
        """
        assert np.issubdtype(type(ind), int)
        if ind == -1:
            return self.parent
        return self.parent[ind]

    def get_left(self, ind):
        """
        Gibt den Index des linken Kindes des durch 'ind' referenzierten Knotens zurück.
        """
        assert np.issubdtype(type(ind), int)
        return self.children[ind][0]

    def get_right(self, ind):
        """
        Gibt den Index des rechten Kindes des durch 'ind' referenzierten Knotens zurück.
        """
        assert np.issubdtype(type(ind), int)
        return self.children[ind][1]

    def get_children(self, ind=-1):
        """
        Gibt die Indizes der Kinder des durch 'ind' referenzierten Knotens zurück.
        """
        assert np.issubdtype(type(ind), int)
        if ind == -1:
            return self.children
        else:
            return self.children[ind]

    def is_leaf(self, ind):
        """
        Gibt zurück, ob es sich beim durch 'ind' referenzierten Knoten um ein Blatt handelt.
        """
        assert np.issubdtype(type(ind), int)
        assert ind >= 0
        if np.all(self.children[ind] == -1):
            return True
        else:
            return False

    def get_nodes(self):
        """
        Gibt die Indizes alles Knoten zurück.
        """
        nodes = [0]
        for i in range(len(self.children)):
            ch = list(self.children[i])
            if ch != [-1, -1]:
                nodes += ch
        return np.array(sorted(nodes))

    def is_root(self, ind):
        """
        Gibt zurück, ob es sich bei dem durch 'ind' referenzierten Knoten um die Wurzel handelt.
        """
        assert np.issubdtype(type(ind), int)
        if ind == 0:
            return True
        else:
            return False

    def get_nr_nodes(self):
        """
        Gibt die Anzahl der Knoten zurück.
        """
        return self.nr_nodes

    def get_nr_dims(self):
        """
        Gibt die Anzahl der repräsentierten Dimensionen zurück.
        """
        return self.nr_dims

    def get_nodes_dfs(self, ind):
        """
        Gibt die Indizes der Knoten des durch 'ind' referenzierten Knotens in dfs Reihenfolge zurück.
        """
        assert np.issubdtype(type(ind), int)
        if self.is_leaf(ind):
            return [ind]
        else:
            left = self.get_left(ind)
            right = self.get_right(ind)
            left_anc = list(self.get_nodes_dfs(left))
            right_anc = list(self.get_nodes_dfs(right))
            return np.array([ind] + left_anc + right_anc)

    def get_sibling(self, ind):
        """
        Gibt den Index des durch 'ind' referenzierten Knotens zurück.
        """
        assert isinstance(ind, int) or isinstance(ind, np.int32)
        p = self.get_parent(ind)
        pc = self.get_children(p)
        if pc[0] == ind:
            return pc[1]
        else:
            return pc[0]

    def is_left(self, ind):
        """
        Gibt zurück, ob der durch 'ind' referenzierte Knoten ein linkes Kind ist.
        """
        assert isinstance(ind, int) or isinstance(ind, np.int32)
        p = self.get_parent(ind)
        pc = self.get_children(p)
        if pc[0] == ind:
            return True
        else:
            return False

    def is_right(self, ind):
        """
        Gibt zurück, ob der durch 'ind' referenzierte Knoten ein rechtes Kind ist.
        """
        assert isinstance(ind, int) or isinstance(ind, np.int32)
        p = self.get_parent(ind)
        pc = self.get_children(p)
        if pc[1] == ind:
            return True
        else:
            return False

    def is_inner_node(self, ind):
        """
        Gibt zurück, ob der durch 'ind' referenzierte Knoten ein innerer Knoten ist.
        """
        assert isinstance(ind, int) or isinstance(ind, np.int32)
        if not np.all(self.children[ind] == [-1, -1]):
            return True
        else:
            return False

    def get_dim(self, ind):
        """
        Gibt die Dimension(en) zurück, die durch den von 'ind' referenzierten Knoten repräsentiert werden.
        """
        assert np.issubdtype(type(ind), int)
        if self.is_leaf(ind):
            array = np.array(self.dim2ind)
            dim = np.argmax(array == ind)
            return np.array([dim])
        else:
            dim_left = list(self.get_dim(self.children[ind][0]))
            dim_right = list(self.get_dim(self.children[ind][1]))
            dim = dim_left + dim_right
            return np.array(dim)

    def get_ind(self, dim):
        """
        Gibt den Index jenes Knotens zurück, der die in 'dim' enthaltenen Dimensionen repräsentiert.
        """
        if not isinstance(dim, np.ndarray) and not isinstance(dim, tuple) \
                and not isinstance(dim, list):
            return self.dim2ind[dim]
        else:
            nodes = self.get_nodes()
            for ind in nodes:
                d = self.get_dim(ind)
                if np.all(d == dim):
                    return ind

    def get_level(self, ind):
        """
        Gibt das Level, auf dem sich der durch 'ind' repräsentierte Knoten befindet, zurück.
        """
        assert np.issubdtype(type(ind), int)
        return self.level[ind]

    def get_nodes_of_level(self, lvl):
        """
        Gibt alle Indizes der Knoten des Level 'lvl' zurück.
        """
        assert np.issubdtype(type(lvl), int)
        nodes = []
        for ind in self.get_nodes():
            if self.get_level(ind) == lvl:
                nodes += [ind]
        return np.array(nodes)

    def get_inner_nodes(self):
        """
        Gibt die Indizes aller innerer Knoten zurück.
        """
        nodes = []
        for ind in self.get_nodes():
            if not self.is_leaf(ind):
                nodes += [ind]
        return np.array(nodes)

    def get_leaves(self):
        """
        Gibt die Indizes aller Blattknoten zurück.
        """
        return self.leaves

    def get_depth(self):
        """
        Gibt die Tiefe des Dimensionsbaums zurück.
        """
        level = np.array(self.level)
        depth = np.max(level)
        return depth

    def get_subtree(self, ind):
        """
        Gibt den Subtree des durch 'ind' referenzierten Knotens zurück.
        """
        assert np.issubdtype(type(ind), int)
        nodes = [ind]
        descendants = [ind]
        while len(nodes) > 0:
            node = nodes.pop(0)
            if not self.is_leaf(node):
                nodes += list(self.get_children(node))
                descendants += list(self.get_children(node))
        return descendants


def equal(dt1, dt2):
    """
    Berechnet, ob die beiden Dimensionsbäume 'dt1' und 'dt2' übereinstimmen.
    @param dt1: tensor.utils.dimtree.dimtree
    @param dt2: tensor.utils.dimtree.dimtree
    @return: bool
    """
    if not np.array_equal(dt1.get_children(), dt2.get_children()):
        return False
    elif not np.array_equal(dt1.get_dim2ind(), dt2.get_dim2ind()):
        return False
    else:
        return True
