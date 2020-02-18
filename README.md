# Bipartite Probabilistic Matching
Shoval Frydman, Kfir Salomon, Itay Levinas

## Problem Description
Given a weighted bipartite graph, we calculate a probability matrix ***P***, such that ***P[i, j]*** represents the probability for the 
vertex ***i*** to match the vertex ***j***.
Note that this matrix is normalized over the rows only. 

## Suggested Algorithms
We suggest three algorithms (null model, flow and updating by degree) to solve this problem. 
The algorithms are implemented in the code directory (the flow algorithm has two implementations - analytic and numeric).

## How to Run
Our main file is *matching_solutions.py*. It includes a class that runs the requested algorithm on a given graph.
This class includes measurements for performance of the algorithm on the graph, based on a given ground truth, and a visualization of the 
graph with its results.

For more information, please refer to the [Wiki](../../wiki) of this repository. 
