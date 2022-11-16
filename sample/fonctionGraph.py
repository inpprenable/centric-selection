import networkx as nx
import numpy as np


def additive_weight_from_node(graph: list, i: int, matrix_weight: np.ndarray) -> float:
    weight = 0
    for j in graph:
        weight += matrix_weight[j, i]
    return weight


def generate_matrix(nb_node: int) -> np.ndarray:
    delay = np.random.rand(nb_node) / 2
    matrix = np.zeros((nb_node, nb_node))
    for i in range(nb_node):
        for j in range(i + 1, nb_node):
            matrix[i, j] = delay[i] + delay[j]
            matrix[j, i] = matrix[i, j]
    return matrix


def generate_node(height: float, width: float, nb_node: int) -> dict:
    list_node = {}
    for i in range(nb_node):
        list_node[i] = (np.random.rand() * height, np.random.rand() * width)
    return list_node


def calcul_distance_two_nodes(nodeA: tuple, nodeB: tuple) -> float:
    return np.sqrt(np.square(nodeA[0] - nodeB[0]) + np.square(nodeA[1] - nodeB[1]))


def calcul_matrice_adjacente(dict_node: dict) -> np.ndarray:
    nb_node = len(dict_node)
    matrix = np.zeros((nb_node, nb_node), dtype=int)
    for i in range(nb_node):
        for j in range(i + 1, nb_node):
            matrix[i, j] = int(calcul_distance_two_nodes(dict_node[i], dict_node[j]))
            matrix[j, i] = matrix[i, j]
    return matrix


def get_list_weight(validators: list, matrix: np.ndarray) -> np.ndarray:
    N = len(validators)
    list_poid = np.zeros(N * (N - 1) // 2, dtype=int)
    indice = 0
    for i in range(N):
        node = validators[i]
        for j in range(i + 1, N):
            adj_node = validators[j]
            list_poid[indice] = (matrix[node, adj_node])
            indice += 1
    return list_poid


def generate_graph_from_pos(list_node: dict) -> nx.Graph:
    G = nx.Graph()
    nb_node = len(list_node)
    for i in range(nb_node):
        G.add_node(i, pos=list_node[i])
    return G


def get_weight_graph(list_node, matrix: np.ndarray) -> int:
    weight = 0
    nb_node = len(list_node)
    for i in range(nb_node):
        if list_node[i]:
            for j in range(i + 1, nb_node):
                if list_node[j]:
                    weight += matrix[i, j]
    return weight


def bool_list_to_int(x) -> list:
    solution = []
    for i in range(len(x)):
        if x[i]:
            solution.append(i)
    return solution


def list_int_to_list_bool(nb_node: int, list_val):
    sortie = np.zeros(nb_node, dtype=bool)
    for val in list_val:
        sortie[val] = True
    return sortie
