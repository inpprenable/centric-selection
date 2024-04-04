import argparse
import json
import math
import os.path
import sys
import time

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from fonctionGraph import generate_node, calcul_matrice_adjacente, get_list_weight, generate_graph_from_pos


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a range of parcours with our selection')
    parser.add_argument("--map", type=str, help="the file which stores the map",
                        default=None)
    parser.add_argument("--val", type=int, help="average number of validator in the map", default=25)
    parser.add_argument("--evValPeriod", type=int,
                        help="Period of random change of the number of validator, set to 0 to never change (default)",
                        default=0)
    parser.add_argument('--evValStd', type=int,
                        help="Std of Gaussian random change of the number of validator (default 0)",
                        default=0)
    parser.add_argument("--evMapPeriod", type=int,
                        help="Period of random change of the node position, set to 0 to never change (default)",
                        default=0)
    parser.add_argument('--evMapStd', type=int,
                        help="Std of Gaussian random change of the node position (default 0, avg = 0)",
                        default=0)
    parser.add_argument("--ellipse", type=int, default=100,
                        help="Set the ellipse of iteration before the representation")
    parser.add_argument("--mu", type=float, help="The coefficient in the calculus", default=1)
    parser.add_argument('--verbose', '-v', action='count', default=0)

    parser.add_argument("--gen", type=int, default=100,
                        help="Set the number of generation to show")
    parser.add_argument('--print', default=False, action='store_true', help="print in a gif file called animation.gif")
    args = parser.parse_args()
    return args


def select_center(list_validators: list, matrix: np.ndarray) -> int:
    N = len(matrix)
    # list_average_delay = [sum(delays) for delays in matrix]
    list_average_delay = [sum(delays[list_validators]) for delays in matrix]
    list_potential_center = [node for node in list_validators]
    if len(list_potential_center) == 0:
        list_potential_center = [node for node in range(N) if
                                 (node not in list_validators)]
    center, center_delay = -1, math.inf
    for node in list_potential_center:
        if list_average_delay[node] < center_delay:
            center = node
            center_delay = list_average_delay[node]
    return center


class ConfigSelection:
    def __init__(self, coef_eloignement: float):
        self.coef_eloignement = coef_eloignement

    def func_eloignement(self, t: float) -> float:
        return np.exp(t / self.coef_eloignement)


class Frame:
    def __init__(self, nb_node: int, config: ConfigSelection):
        self.timestamp = 0
        self.config = config
        self.nb_node = nb_node
        self.nb_val = nb_node
        self.past_validator = np.zeros(nb_node, dtype=np.float128)
        self.validators = list(np.arange(nb_node))

    def select_new_val(self, center: int, nb_val: int, matrix: np.ndarray,
                       past_validators: np.ndarray) -> list:
        N = len(matrix)
        delays = np.zeros(N)
        for i in range(N):
            delays[i] = matrix[center, i] * self.config.func_eloignement(past_validators[i])
        potential_validators = np.arange(N)
        potential_validators_sorted = [x for _, x in sorted(zip(delays, potential_validators))]
        return potential_validators_sorted[: nb_val]

    def update(self, matrix: np.ndarray, nb_val: int):
        self.timestamp += 1
        N = self.nb_node
        self.nb_val = nb_val
        self.center = select_center(self.validators, matrix)
        self.validators = self.select_new_val(self.center, nb_val, matrix, self.past_validator)
        self.uptade_past(N)

    def uptade_past(self, N: int):
        for i in range(N):
            if i in self.validators:
                self.past_validator[i] += 1 / self.nb_val
            else:
                self.past_validator[i] -= 1 / (self.nb_node - self.nb_val)

    # Return the average weight, the median weight
    def calcul_metric(self, matrix: np.ndarray) -> tuple:
        list_weight = get_list_weight(self.validators, matrix)
        return np.average(list_weight), np.median(list_weight), np.max(list_weight)

    def print(self):
        print("On epoch {}, nb_val = {}, center = {}\n"
              "validators = {}".format(self.timestamp, self.nb_val, self.center, self.validators))

    def gen_color_map(self) -> list:
        color_map = []
        for node in range(self.nb_node):
            if hasattr(self, 'center') and node == self.center:
                color_map.append("orange")
            elif node in self.validators:
                color_map.append("red")
            else:
                color_map.append("blue")
        return color_map
    # def show(self):


class Matrix:
    def __init__(self, height: float, width: float, nb_node: int, evol_mat: int, variation: float):
        self.init_list_node = generate_node(height, width, nb_node)
        self.current_list_node = self.init_list_node.copy()
        self.init_matrix = calcul_matrice_adjacente(self.init_list_node)
        self.current_matrix = self.init_matrix
        self.evol_mat = evol_mat
        self.counter = 0
        self.variation = variation

    def update(self) -> bool:
        if self.evol_mat != 0 and self.counter == 0:
            self.counter = (self.counter + 1) % self.evol_mat
            list_variation = np.random.normal(0, self.variation, len(self.init_list_node))
            self.current_list_node = self.init_list_node.copy()
            for i in range(len(list_variation)):
                self.current_list_node[i] += list_variation[i]
            self.current_matrix = calcul_matrice_adjacente(self.current_list_node)
            return True
        return False

    @staticmethod
    def create_from_file(filename: str, evol_mat: int, variation: float):
        if filename == "" or not os.path.exists(filename):
            raise Exception("The file doesn't exist")
        with open(filename, "r") as file:
            list_node = json.loads(file.read())
            list_node = {int(k): v for k, v in list_node.items()}
        list_node_mat = Matrix(100, 100, 100, evol_mat, variation)
        list_node_mat.init_list_node = list_node
        list_node_mat.current_list_node = list_node_mat.init_list_node.copy()
        list_node_mat.init_matrix = calcul_matrice_adjacente(list_node_mat.init_list_node)
        list_node_mat.current_matrix = list_node_mat.init_matrix
        return list_node_mat


def calc_new_nb_val(nb_node: int, nb_val_init: int, variation_val: float) -> int:
    new_nb_val = nb_val_init + int(np.random.normal(0, variation_val, 1))
    if new_nb_val < 4:
        return 4
    if new_nb_val > nb_node:
        return nb_node
    return new_nb_val


if __name__ == '__main__':
    args = create_parser()

    height, width = 100, 100
    nb_node, nb_val_0 = 200, args.val
    evol_val, variation_val = args.evValPeriod, args.evValStd
    evol_mat, variation_mat = args.evMapPeriod, args.evMapStd

    liste_node_matrix = None
    if args.map is not None:
        if os.path.isfile(args.map):
            liste_node_matrix = Matrix.create_from_file(args.map, evol_mat, variation_mat)
            nb_node = len(liste_node_matrix.init_list_node)
        else:
            print("The map file doesn't exist", file=sys.stderr)
            exit(1)
    else:
        liste_node_matrix = Matrix(height, width, nb_node, evol_mat, variation_mat)
    config = ConfigSelection(args.mu)
    frame = Frame(nb_node, config)
    nb_val = calc_new_nb_val(nb_node, nb_val_0, variation_val)
    counter_val = 0
    elipse = args.ellipse

    t0 = time.time()
    for i in range(elipse):
        if evol_val := 0:
            counter_val = (counter_val + 1) % evol_val
            if counter_val == 0:
                nb_val = calc_new_nb_val(nb_node, nb_val_0, variation_val)
        liste_node_matrix.update()
        frame.update(liste_node_matrix.current_matrix, nb_val)
    if elipse > 0:
        print("Ellipse done in {:.2f} seconds per iteration".format((time.time() - t0) / elipse))

    fig, ax = plt.subplots()
    G = generate_graph_from_pos(liste_node_matrix.current_list_node)
    color_map = frame.gen_color_map()
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color=color_map, with_labels=True)


    def update(num, ax, _):
        global G, counter_val, nb_val
        ax.clear()
        t0 = time.time()
        if evol_val != 0:
            counter_val = (counter_val + 1) % evol_val
            if counter_val == 0:
                nb_val = calc_new_nb_val(nb_node, nb_val_0, variation_val)
                print("New nb_val : ", nb_val)
        changed = liste_node_matrix.update()
        frame.update(liste_node_matrix.current_matrix, nb_val)
        if changed:
            G = generate_graph_from_pos(liste_node_matrix.current_list_node)
        color_map = frame.gen_color_map()
        pos = nx.get_node_attributes(G, 'pos')

        nx.draw(G, pos, node_color=color_map, with_labels=True)


    ani = FuncAnimation(fig, update, frames=args.gen, interval=250, repeat=False, fargs=(ax, 0))
    if args.print:
        ani.save('animation.gif', writer='imagemagick')
    plt.show()
