"""
This algorithm is simple:
From the weight matrix, we produce a matrix p.
For each row in w, we build a row in p, in which p_ij = 0 if w_ij is not maximal in row i, and otherwise
p_ij = 1 / (number of maxima).
"""

import numpy as np
from matching_solutions import MatchingProblem


def algorithm(mp: MatchingProblem, params=None):
    """
    The implementation of the null model algorithm.
    :param mp: The main class containing the graph and its biadjacency matrix, on which we want to apply the algorithm.
    The algorithm uses the biadjacency weight matrix only.
    :param params: We will not use it anyway
    :return: The final probability matrix p.
    """
    p = np.zeros_like(mp.w)
    for i in range(p.shape[0]):
        p[i, :] = np.where(mp.w[i, :] == max(mp.w[i, :]), 1, 0)
        s = np.sum(p[i, :])
        if s:
            p[i, :] = np.divide(p[i, :], s)
    return p
