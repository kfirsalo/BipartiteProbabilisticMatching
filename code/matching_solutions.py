import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class MatchingProblem:
    algorithm_names = {
        "deg_update": "Updating by degree",
        "flow_analytic": "Flow - analytic solution",
        "flow_numeric": "Flow - numeric solution",
        "null_model": "Null model"
    }

    def __init__(self, graph_path, matches_path, algorithm, params):
        """
        The main class. Receives a graph and the ground truth values and implements the requested algorithm.
        The results can later be measured using top-k accuracy and the sum of probabilities score, or can be visualized.

        :param graph_path: The path to a csv file of the weighted biadjacency matrix.
        :param matches_path: The path to a csv file of the ground truth matches.
        :param algorithm: The name of the desired algorithm. Can be one of the following four:
               "deg_update", "flow_analytic", "flow_numeric" or "null_model".
        :param params: The dictionary of the parameters required for the algorithm.
        """
        self.algorithm_name = self.algorithm_names[algorithm]
        self.graph, self.w = self._load_graph(graph_path)
        self.unw_adj = np.where(self.w != 0, 1, 0)  # Unweighted bipartite adjacency matrix
        self.p_mat = self.algorithm(algorithm, params)
        self.true_matches = self._load_matches(matches_path)

    @staticmethod
    def _load_graph(graph_path):
        """
        Load the bipartite graph and its weighted biadjacency matrix (i.e. of shape |side1| x |side2|)

        :param graph_path: The path to a csv file of the weighted biadjacency matrix.
        :return: A networkx Graph and a numpy array of the bipartite weight matrix.
        """
        biadj_matrix = pd.read_csv(graph_path, header=None).to_numpy()
        graph = nx.algorithms.bipartite.from_biadjacency_matrix(csr_matrix(biadj_matrix))
        return graph, biadj_matrix

    def algorithm(self, algorithm, params):
        """
        Import and implement the requested algorithm, to create a probability matrix of the same dimensions as the
        biadjacency matrix, where the (i, j)-th element represents the probability that the vertex i of the first side
        matches the vertex j of the second side.
        Note that p is normalized row-wise (i.e. sum_j (p_ij) = 1), but not necessarily column-wise.

        :param algorithm: The string indicating which algorithm to run. Can be one of the following four:
               "deg_update", "flow_analytic", "flow_numeric" or "null_model".
        :param params: The dictionary of the parameters required for the algorithm.
        :return: The final probability matrix p
        """
        if algorithm == "deg_update":
            from updating_by_degree import algorithm
        elif algorithm == "flow_analytic":
            from flow_analytic import algorithm
        elif algorithm == "flow_numeric":
            from flow_numeric import algorithm
        else:  # "null_model"
            from null_model import algorithm
        p = algorithm(self, params)
        return p

    @staticmethod
    def _load_matches(matches_path):
        """
        Load the matches file, as a numpy array where the first column represents the vertices of one side and the
        second column represents the matching (ground truth) vertices from the other side.
        NOTE: The toy models we created had vertices with indices 1, 2, ..., number_of_vertices_per_side.
        Therefore, we subtract 1 from the loaded file. To use this function as is, the vertices should be indexed
        accordingly.

        :param matches_path: The path to a csv file of the ground truth matches.
        :return: A numpy array of the matches.
        """
        # NOTE: Yoram's toy models are with vertices 1 to 100. If another format is used, generalize the function. 
        return pd.read_csv(matches_path, header=None).to_numpy() - 1

    def nonzero_degree_vertices(self):
        """
        Calculate and return the vertices with degree > 0. We can calculate the probability to them only.
        """
        return [v for v in range(self.unw_adj.shape[0]) if np.sum(self.unw_adj[v, :]) > 0]

    def top_k_accuracy(self, k):
        """
        Calculate the top-k accuracy.
        Here, top-k accuracy means taking the number of vertices from the first side for which the ground truth vertex
        appears in the top k vertices from the second side by the probability matrix, over the total number of vertices
        from the first side.

        :param k: int. The smaller k is, the more difficult it is to reach high accuracies.
        :return: The accuracy.
        """
        correct = 0.
        tried = 0.
        for v in range(self.p_mat.shape[0]):
            true = self.true_matches[v, 1]
            preds = np.argsort(- self.p_mat[v, :])[:k]
            if true in preds:
                correct += 1
            tried += 1
        return correct / tried

    def sum_prob_score(self):
        """
        The sum of probabilities score, i.e. sum_i (p_[i, t]),
        where t is the index of the ground truth vertex and p is the probability matrix

        :return:
        """
        return sum([self.p_mat[i, self.true_matches[i, 1]] for i in range(self.true_matches.shape[0])])

    def visualize_results(self, saving_path):
        """
        Create and save a figure of the bipartite graph, with colored edges:
        Green edges represent true positive edges, blue represent true negative, orange represents false positive and
        red represents false negative.

        :param saving_path: The path in which the figure will be saved.
        """
        left_side_vertices = self.true_matches[:, 0]
        pos = nx.bipartite_layout(self.graph, left_side_vertices)
        edge_colors = []
        counts = [0, 0, 0, 0]
        for e in self.graph.edges:
            source = e[0]
            target = e[1] - self.true_matches.shape[0]  # The full-graph index -> index in the graph's second side.
            pred = np.argmax(self.p_mat[source, :])
            if self.true_matches[source, 1] == target:
                if pred == target:
                    # TP
                    edge_colors.append('g')
                    counts[0] += 1
                else:
                    # FN
                    edge_colors.append('r')
                    counts[3] += 1
            else:
                if pred == target:
                    # FP
                    edge_colors.append('orange')
                    counts[2] += 1
                else:
                    # TN
                    edge_colors.append('b')
                    counts[1] += 1
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(self.graph, pos)
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors)
        plt.title(self.algorithm_name)
        plt.text(0, -0.75, 'TP (green) - %d, TN (blue) - %d, FP (orange) - %d, FN (red) - %d' %
                 (counts[0], counts[1], counts[2], counts[3]), fontsize=12,
                 horizontalalignment='center', verticalalignment='center',
                 bbox={'facecolor': 'grey', 'alpha': 0.7})
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.savefig(saving_path)


if __name__ == '__main__':
    import os

    parameters = {"rho_0": 0.3, "rho_1": 0.6, "epsilon": 1e-2}
    mp = MatchingProblem(os.path.join("graphs", "Obs_Usage_Exp_4.csv"), os.path.join("graphs", "Real_Tags_Exp_4.csv"),
                         "flow_numeric", parameters)
    mp.visualize_results("visualization_example.png")
