from tensor.utils.dimtree import equal
import numpy as np


def gramians_sum(cls, summanden):
    """
    Berechnet die reduzierten Gram'schen Matrizen für eine implizite Summe von tensor.htucker.htucker Objekten.
    Die Summanden werden als Liste übergeben und wurden noch nicht aufaddiert.
    Hinweis: Die hierarchischen Tuckertensoren in 'summanden' müssen identische Dimensionsbäume besitzen.
    @param summanden: Liste bestehnd aus tensor.htucker.htucker Objekten
    @param copy: bool
    @return: dict
    """

    # Argument checks
    if not isinstance(summanden, list):
        raise TypeError("'summanden' ist keine Liste.")
    if not all(isinstance(item, cls) for item in summanden):
        raise ValueError("'summanden' ist keine List mit hierarchischen Tuckertensoren als Elementen.")
    if not all(item.shape == summanden[0].shape for item in summanden):
        raise ValueError("Die hierarchischen Tuckertensoren in 'summanden' haben nicht alle dieselben Modi.")
    if not all(equal(item.dtree, summanden[0].dtree) for item in summanden):
        raise ValueError("Die hierarchischen Tuckertensoren in 'summanden' haben nicht alle identische"
                         " Dimensionsbäume.")
    
    U = {}
    M = {}

    # Referenz-Dimensionsbaum
    dtree = summanden[0].dtree
    
    # Berechne M[t] = U[t].T @ U[t]
    # Durschreite den Baum von den Blättern hin zur Wurzel
    for level in range(dtree.get_depth(), 0, -1):
        for t in dtree.get_nodes_of_level(level):
            
            if dtree.is_leaf(t):
                # t ist Blattknoten
                # Konkateniere die t-Blattmatrizen aller Summanden
                U[t] = np.hstack([item.U[t] for item in summanden])
                
                # Berechne alle Paare der Art summanden[i].U[t].T @ summanden[j].U[t]
                Mt = U[t].T @ U[t]
                
                # Für einfacheren Zugriff, baue ein dict M[t]
                # mit M[t][i,j] = summanden[i].U[t].T @ summanden[j].U[t]
                M[t] = {}
                ranks = [item.U[t].shape[1] for item in summanden]
                cum_ranks = np.cumsum([0] + ranks)
                for i in range(len(summanden)):
                    for j in range(len(summanden)):
                        M[t][i, j] = Mt[cum_ranks[i]:cum_ranks[i + 1], cum_ranks[j]:cum_ranks[j + 1]]
            else:
                # t ist innerer Knoten
                # Kinder von t
                r, l = dtree.get_right(t), dtree.get_left(t)
                
                # Baue dict mit Bt[i] = summanden[i].B[t]
                # (Bt[i] entspricht dem i-ten Diagonalblock des Transfertensors
                # der Summe)
                Bt = {i: summanden[i].B[t] for i in range(len(summanden))}
                
                # Berechne M[t]
                M[t] = {}
                for kk in range(len(summanden)):
                    for ll in range(len(summanden)):
                        Mlbt = np.tensordot(M[l][ll,kk], Bt[kk], axes=[1,0])
                        Mrbt = np.tensordot(M[r][kk,ll], Bt[ll], axes=[1,1])
                        M[t][kk,ll] = np.tensordot(Mlbt, Mrbt, axes=[[0,1], [1,0]])

    # Gram'sche Matrix der Wurzel
    G = {0: np.ones((1, 1))}
    # Root children
    rr, rl = dtree.get_right(0), dtree.get_left(0)
    # Berechne <B[0], B[0] x_0 G[0]> _(0,1) and _(0,3)
    G[rl], G[rr] = {}, {}
    # Baue dict mit Bt[u] = summanden[i].B[0]
    # (Bt[i] entspricht dem i-ten Diagonalblock des Transfertensors
    # der Summe)
    Bt = {i: summanden[i].B[0] for i in range(len(summanden))}
    for kk in range(len(summanden)):
        for ll in range(len(summanden)):
            Mlbt = np.tensordot(M[rl][kk,ll], Bt[ll], axes=[1,0])
            Mrbt = np.tensordot(M[rr][kk,ll], Bt[ll], axes=[1,1])
            G[rl][kk,ll] = np.tensordot(Bt[kk], Mrbt, axes=[[1,2], [0,2]])
            G[rr][kk,ll] = np.tensordot(Bt[kk], Mlbt, axes=[[0,2], [0,2]])

    # Berechne von der Wurzel zu den Blättern die Gram'schen Matrizen
    for level in range(1, dtree.get_depth(), 1):
        for t in dtree.get_nodes_of_level(level):
            if dtree.is_leaf(t):
                continue

            # Kinder
            l, r = dtree.get_left(t), dtree.get_right(t)

            B = {jj: summanden[jj].B[t] for jj in range(len(summanden))}

            G[l] = {}
            G[r] = {}
            for kk in range(len(summanden)):
                for ll in range(len(summanden)):
                    B_mod_klk = np.tensordot(B[kk],G[t][ll,kk], axes=[2,1])
                    B_r_kll = np.tensordot(M[l][kk, ll], B[ll], axes=[1,0])
                    B_l_kll = np.tensordot(M[r][kk, ll], B[ll], axes=[1,1])
                    B_l_kll = np.swapaxes(B_l_kll, 0,1)

                    G[l][kk, ll] = np.tensordot(B_mod_klk, B_l_kll, axes=[[1, 2], [1, 2]])
                    G[r][kk, ll] = np.tensordot(B_mod_klk, B_r_kll, axes=[[0, 2], [0, 2]])

    for t in dtree.get_nodes():
        if t == 0:
            continue
        G[t] = np.block([[G[t][kk,ll] for ll in range(len(summanden))] for kk in range(len(summanden))])
    return G



