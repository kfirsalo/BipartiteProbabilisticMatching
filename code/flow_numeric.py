"""
An numeric algorithm for solving the matching problem, using the following algorithm:
Input: A weighted bipartite graph.

1. Let W be the ("full") weighted adjacency matrix of the bipartite graph.
   W will be normalized and represent a directed graph, such that for every vertex with output degree >= 1
   the sum of all weights of edges from it will be 1.
2. Let p be a vector of the amount of fluids held in each vertex. It is initialized as a vector of ones, of size equal
   to the number of vertices.
3. The vector p is iteratively updated until convergence (i.e., the 1-norm of the step, p_new - p_old, is smaller than a
   parameter epsilon). The step of updating is done by flowing on the edges according to the following formula:
                        p = rho_0 * p + rho_1 (p * w) + (1 - rho_0 - rho_1) ones_vector
   where rho_0, rho_1 are parameters of the model, ones_vector is a vector of ones with the same shape as p.
4. Using p and W, we create a matrix of contributions, where the (i, j)-th element represents the amount of fluid that
   j receives from i per time unit, according to the converged vector p.
5. The contributions matrix is normalized like in part 1, resulting the probabilities matrix.
   Here, we take the biadjacency matrix which is the upper right block for calculating the desired measurements.
   However, the entire (non-symmetric) matrix is calculated.
Output: The required probabilities matrix.


NOTE: The function "algorithm", which is the main function of this algorithm, receives a dictionary named 'params'
of the following parameters (e.g. {'rho_0': 0.6, ...}):
    rho_0 - Controls the amount of fluid we choose to keep in each vertex (rather than flow to its neighbors) per time
            unit.
    rho_1 - Controls the amount of fluid we choose to pass on the edges. (1 - rho_0 - rho_1) represents the amount of
            fluid per time unit added to each vertex.
    epsilon - The tolerance constant. Controls how close we want to converge to the analytic solution.
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
    w = nx.to_numpy_array(mp.graph)
    w = normalization(w)
    p_node_balance = node_balance(w, params['rho_0'], params['rho_1'], params['epsilon'])
    contribution_matrix = calculate_contribution_matrix(p_node_balance, w)
    c = normalization(contribution_matrix[:100, 100:])
    return c


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


def node_balance(w, r0, r1, epsilon):
    """
    Create the vector p (initialized as a vector of ones) and run the flow until convergence.

    :param w: The normalized full adjacency matrix.
    :param r0: The parameter rho_0.
    :param r1: The parameter rho_1.
    :param epsilon: The tolerance parameter. We will stop the iterations if ||p_new - p_old||_1 < epsilon.
    :return: The final (steady state) vector p.
    """
    p = np.ones(w.shape[0])
    condition = epsilon + 1
    while condition > epsilon:
        p_new = r0 * p + r1 * np.dot(p, w) + (1 - r0 - r1) * np.ones_like(p)
        condition = np.linalg.norm(p_new-p, 1)
        p = p_new
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
            w[i, j] = p[i] * w[i, j] / p[j]
    return w




