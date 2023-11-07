import numpy as np

def neumann_diff_matrix(n):
    """
    n: number of entries
    """
    A =  []
    
    # First row of A
    first_row = np.zeros(n)
    first_row[0] = -(7/2)
    first_row[1] = 4
    first_row[2] = -(1/2)
    A += [first_row]
    
    # Inner rows of A
    for i in range(1,n-1):
        row = np.zeros(n)
        row[i-1] = 1
        row[i] = -2
        row[i+1] = 1
        A += [row]
        
    # Last row of A
    last_row = np.zeros(n)
    last_row[-1] = -(7/2)
    last_row[-2] = 4
    last_row[-3] = -(1/2)
    A += [last_row]
    
    return np.array(A)