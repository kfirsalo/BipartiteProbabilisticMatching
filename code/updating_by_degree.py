"""
An algorithm for solving the matching problem, using the following algorithm:
Input: A weighted bipartite graph.

1. Initialize p as a softmax normalization of the biadjacency weight matrix.
2. Let max_deg be the maximal degree of a vertex in the graph.
3. For i=1, 2, ..., max_deg:
        Do num_updates_per_deg times:
            Do on the matrix forward (from one side to the other) and then backwards:
                p_tilde = normalize the rows/columns (depends on the direction) of p, which correspond to vertices with
                          degree larger than or equal to i. The normalization: sum_j (p_ij) = 1
                          (or i if normalizing columns).
                new p = (1 - eta_0 - eta_1) * p + eta_0 * p_tilde + eta1 * (1 / i for each matrix element that
                                                                            represents an existing edge, otherwise 0).
4. Finally, Normalize the rows of p, as done in 3, on all vertices.
Output: The required probabilities matrix.


NOTE: The function "algorithm", which is the main function of this algorithm, receives a dictionary named 'params'
of the following parameters (e.g. {'alpha': 0.1, ...}):
    num_updates_per_deg - The number of updates of the matrix p, done for each degree (every update includes a forward
    and backward passes over the matrix).
    alpha - Controls the initial softmax normalization.
    eta0 - Controls the step size of updating p towards p_tilde.
    eta1 - Controls the effect of adding (1 / degree) to each vertex with a positive degree we update.
"""
import numpy as np
from matching_solutions import MatchingProblem


def algorithm(mp: MatchingProblem, params):
    """
    The main function of the algorithm.

    :param mp: The main class containing the graph and its biadjacency matrix, on which we want to apply the algorithm.
    The algorithm uses the networkx graph, the biadjacency weight matrix and the unweighted biadjacency matrix only.
    :param params: The dictionary of the parameters of this algorithm.
    :return: The final probability matrix p.
    """
    p = first_normalization(mp.w.copy(), params['alpha'])
    max_deg = max([t[1] for t in mp.graph.degree()])
    for d in range(1, max_deg + 1):
        for num_updates in range(params['num_updates_per_deg']):
            for direction in ['forward', 'backward']:
                p_tilde = normalize(mp, p, d, direction)
                p = update(mp, p, p_tilde, params['eta0'], params['eta1'], d, direction)
    # Normalize by the desired direction to get the final p we want.
    p = normalize(mp, p, 1, 'forward')
    return p


def first_normalization(p, alpha):
    """
    Perform a softmax on the nonzero values in p, to set the orders of magnitude.
    The softmax is controlled by alpha (softmax(x)_i = exp(alpha * x_i) / sum_j(exp(alpha * x_j))).
    We normalize by the rows (assuming that the rows and not the columns are the interesting values here).

    :param p: Initialized p matrix
    :param alpha: A parameter to control the softmax.
    :return: The normalized matrix p.
    """
    p_new = np.zeros_like(p)
    maxima = np.max(p, axis=1)
    for i in range(p_new.shape[0]):
        for j in range(p_new.shape[1]):
            p_new[i, j] = np.exp(alpha * (p[i, j] - maxima[i])) if p[i, j] else 0.
    denominators = np.sum(p_new, axis=1)
    denominators = np.where(denominators == 0, 1., denominators)
    return np.divide(p_new.T, denominators).T


def normalize(mp: MatchingProblem, p, d, direction):
    """
    Normalize the matrix p by rows/columns (direction='forward' or 'backward' respectively).
    The normalization is set such that for all vertices with positive degree, the sum of p over the corresponding
    row/column will become 1.

    :param mp: The main class. Used for finding the vertices on which we work.
    :param p: The matrix p, currently not normalized.
    :param d: The degree, by which we will determine the vertices on which we work.
    :param direction: 'forward' (rows) or 'backward' (columns). Determines what we seek to normalize.
    :return: The normalized p, later named p_tilde
    """
    active_vertices = unfrozen_vertices(mp, d, direction)
    sums = np.array([(np.sum(p[v, :]) if direction == 'forward' else np.sum(p[:, v])) if v in active_vertices else
                     1. for v in range((p.shape[0] if direction == 'forward' else p.shape[1]))])
    sums = np.where(sums == 0, 1, sums)  # Problem in the toy models: vertices with degree zero.
    normed_p = np.divide(p.T, sums).T if direction == 'forward' else np.divide(p, sums)
    return normed_p


def update(mp: MatchingProblem, p, p_tilde, eta0, eta1, d, direction):
    """
    Do the updating step of the matrix p:
    p = (1 - eta_0 - eta_1) * p + eta_0 * p_tilde + eta1 * (1 / i for each matrix element that
    represents an existing edge, otherwise 0)

    :param mp: The main class. Used for finding the vertices on which we work.
    :param p: The matrix p before updating.
    :param p_tilde: The normalized version of p, formerly calculated in the function "normalize".
    :param eta0: Controls the step size of updating p towards p_tilde.
    :param eta1: Controls the effect of adding (1 / degree) to each vertex with a positive degree we update.
    :param d: The degree, by which we will determine the vertices on which we work.
    :param direction: 'forward' (rows) or 'backward' (columns). Determines which side we update.
    :return: The updated matrix, which will become the new p.
    """
    active_vertices = list(unfrozen_vertices(mp, d, direction))
    updated_p = p
    adding_factor = np.where(p > 0, eta1 / float(d), 0.)  # Add only if there is an edge
    if direction == 'forward':
        updated_p[active_vertices, :] = \
            (1 - eta0 - eta1) * p[active_vertices, :] + eta0 * p_tilde[active_vertices, :] + \
            adding_factor[active_vertices, :]
    else:
        updated_p[:, active_vertices] = \
            (1 - eta0 - eta1) * p[:, active_vertices] + eta0 * p_tilde[:, active_vertices] + \
            adding_factor[:, active_vertices]
    return updated_p


def unfrozen_vertices(mp: MatchingProblem, d, direction):
    """
    Find which vertices are relevant for the current task. They are the vertices with degree larger than d,
    otherwise we don't touch them at all.

    :param mp: The main class.
    :param d: The degree that every vertex v must pass (by 1 or more) to be counted for the task.
    :param direction: 'forward' or 'backward'. Determines on which side we work.
    :return: A set of the relevant vertices.
    """
    if direction == 'forward':
        degs = np.sum(mp.unw_adj, axis=1)
    else:
        degs = np.sum(mp.unw_adj, axis=0)
    return {v for v in range(len(degs)) if degs[v] >= d}
