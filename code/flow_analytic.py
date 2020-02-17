"""
An algorithm for solving the matching problem, using the following algorithm:
Input: A weighted bipartite graph.

1. Let W be the ("full") weighted adjacency matrix of the bipartite graph.
   W will be normalized and represent a directed graph, such that for every vertex with output degree >= 1
   the sum of all weights of edges from it will be 1.
2. Let p be a vector of the amount of fluids held in each vertex. See the full statement of this problem in the
   documentation of the numeric solution.
   We will calculate the vector p that should be in a steady state:
   p = (1 - rho_0 - rho_1) ones_vector * ( (1-rho_0) I - rho_1 W) ^ -1

   Where rho_0, rho_1 are parameters of the model, ones_vector is a vector of ones, shaped as p,
   I is the identity matrix (of same dimensions as W) and ^ -1 means taking the inverse of the matrix.
3. Using p and W, we create a matrix of contributions, where the (i, j)-th element represents the amount of fluid that
   j receives from i per time unit, in the steady state.
4. The contributions matrix is normalized like in part 1, resulting the probabilities matrix.
   Here, we take the biadjacency matrix which is the upper right block for calculating the desired measurements.
   However, the entire (non-symmetric) matrix is calculated.
Output: The required probabilities matrix.


NOTE: The function "algorithm", which is the main function of this algorithm, receives a dictionary named 'params'
of the following parameters (e.g. {'rho_0': 0.1, ...}):
    rho_0 - Controls the amount of fluid we choose to keep in each vertex (rather than flow to its neighbors) per time
            unit.
    rho_1 - Controls the amount of fluid we choose to pass on the edges. (1 - rho_0 - rho_1) represents the amount of
            fluid per time unit added to each vertex.
"""

import networkx as nx
import numpy as np
from matching_solutions import MatchingProblem


def algorithm(mp: MatchingProblem, params):
    """
    The main function of the algorithm.

    :param mp: The main class containing the graph on which we want to apply the algorithm.
    :param params: The dictionary of the parameters of this algorithm.
    :return: The final probability matrix.
    """
    w = create_full_normalize_adj(mp.graph)
    p = calculate_p(w, params['rho_0'], params['rho_1'])
    contribution_matrix = calculate_contribution_matrix(p, w)
    c = normalization(contribution_matrix[:100, 100:])
    return c


def create_full_normalize_adj(graph):
    """
    From the given networkx bipartite graph,
    create the full adjacency matrix and normalize it by rows.

    :param graph: Our bipartite graph.
    :return: The normalized full adjacency matrix of the graph.
    """
    m = np.array(nx.to_numpy_matrix(graph))
    w = normalization(m)
    return w


def normalization(m):
    """
    Normalize the matrix by rows. The normalization is set such that for all vertices with positive degree (meaning the
    sum of the row is not zero), the sum of m over the corresponding row will become 1.

    :param m: The adjacency matrix.
    :return: The normalized adjacency matrix by rows.
    """
    sums = np.array([np.sum(m[v, :]) if np.sum(m[v, :]) else 1. for v in range(m.shape[0])])
    m = np.divide(m.T, sums).T
    return m


def calculate_p(w, r0, r1):
    """
    Calculate the vector p analytically, as explained above.

    :param w: The normalized full adjacency matrix.
    :param r0: The parameter rho_0.
    :param r1: The parameter rho_1.
    :return: The vector p of amounts of fluid on each vertex in steady state.
    """
    a = (1-r0) * np.identity(w.shape[0]) - r1 * w
    inverse = np.linalg.inv(a)
    ones_vector = np.ones(inverse.shape[0]).reshape((1, w.shape[0]))
    p = np.matmul((1-r0-r1) * ones_vector, inverse)
    return p


def calculate_contribution_matrix(p, w):
    """
    Using p vector and w, we calculate the contribution matrix: c_ij = p_i * w_ij / p_j
    where c_ij is the contribution of vertex i to the vertex j, as explained above.

    :param p: The vector p of amounts of fluids in steady state.
    :param w: The normalized full adjacency matrix
    :return: The contribution matrix
    """
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i, j] = p[0, i] * w[i, j] / p[0, j]
    return w
