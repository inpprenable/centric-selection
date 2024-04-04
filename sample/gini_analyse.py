import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from sample.center_graphical import Frame, Matrix, ConfigSelection
from sample.center_scan import gini_coefficient_worst


def gini_coefficient(array: np.ndarray) -> float:
    """Calculate the Gini coefficient of a numpy array."""
    # extradited from: https://github.com/oliviaguest/gini.git
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a range of parcours with our selection')
    parser.add_argument("input", type=str, help="The position node file")
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("--min", type=int, default=4,
                        help="Set the minimal number of validator for the research")
    parser.add_argument("--max", type=int, default=math.inf,
                        help="Set the maximal number of validator for the research")
    parser.add_argument("--step", type=int, default=1,
                        help="Set the step in the loop of number of validator for the research")
    parser.add_argument("--mu", type=float, help="The coefficient in the calculus", default=1)
    parser.add_argument("--elipse", type=int, default=10,
                        help="Set the elipse of iteration before the sampling")
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--gen", type=int, default=200,
                        help="Set the number of generation")
    parser.add_argument("--val", type=int, help="The number of validator", default=-1)
    parser.add_argument("--ellipse", type=int, default=0,
                        help="Set the elipse of iteration before the sampling")
    args = parser.parse_args()
    return args


class Metric:
    def __init__(self, nb_node: int):
        self.nb_node = nb_node
        self.timestamp = 0
        self.history = []

    def update_frame(self, frame: Frame, matrix: np.ndarray):
        self.prev_val = frame.validators.copy()
        self.timestamp += 1
        self.history.append(frame.validators)

    def compute_sub_gini(self, sub_history: list, horizon: int) -> float:
        validator_distribution = np.zeros(self.nb_node, dtype=float)
        if len(sub_history) < horizon:
            validator_distribution = np.ones(self.nb_node, dtype=float) * (horizon - len(sub_history))
        for action in sub_history:
            weight = self.nb_node / len(action)
            for node in action:
                validator_distribution[node] += weight
        return gini_coefficient(validator_distribution)

    def compute_gini_on_run(self, horizon: int) -> np.ndarray:
        gini_list = np.zeros(self.timestamp, dtype=float)

        for i in range(len(gini_list)):
            sub_history = self.history[max(0, i - horizon): i]
            nb_val = len(self.history[i])
            gini_list[i] = self.compute_sub_gini(sub_history, horizon) / (
                1 if nb_val == self.nb_node else gini_coefficient_worst(nb_val, self.nb_node))
        return gini_list


if __name__ == '__main__':
    args = create_parser()
    liste_node_matrix = Matrix.create_from_file(args.input, 0, 0)
    nb_node = len(liste_node_matrix.init_list_node)
    min_val = args.min
    max_val = min(args.max, nb_node)
    step = args.step

    list_horizon = [nb_node, 2 * nb_node, 3 * nb_node, 4 * nb_node, 5 * nb_node, 6 * nb_node, 7 * nb_node, 8 * nb_node,
                    9 * nb_node]

    if args.val != -1:
        nb_val = args.val
        config = ConfigSelection(args.mu)
        frame = Frame(nb_node, config)
        metric = Metric(nb_node)

        for i in range(args.ellipse):
            frame.update(liste_node_matrix.current_matrix, nb_val)

        for i in range(args.gen):
            frame.update(liste_node_matrix.current_matrix, nb_val)
            metric.update_frame(frame, liste_node_matrix.current_matrix)

        print(f"Final Gini index : {metric.compute_sub_gini(metric.history, metric.timestamp)}")
        plt.figure()
        for horizon in list_horizon:
            plt.plot(metric.compute_gini_on_run(horizon), label=horizon)
        plt.legend()
        plt.title("Relative Gini index according to the horizon size")
        plt.ylabel("Relative Gini index")
        plt.xlabel("Step")
        plt.show()
        exit(0)

    list_max_gini = {}
    list_nb_val = []
    for horizon in list_horizon:
        list_max_gini[horizon] = []

    total = (max_val + 1 - min_val) // step
    for index, nb_val in enumerate(range(min_val, max_val + 1, step)):
        if total > 10 and index % (total // 10) == 0:
            print("{}/100".format(int(100 * index / total)))

        list_nb_val.append(nb_val)

        config = ConfigSelection(args.mu)
        frame = Frame(nb_node, config)
        metric = Metric(nb_node)

        for i in range(args.ellipse):
            frame.update(liste_node_matrix.current_matrix, nb_val)

        for i in range(args.gen):
            frame.update(liste_node_matrix.current_matrix, nb_val)
            metric.update_frame(frame, liste_node_matrix.current_matrix)

        for horizon in list_horizon:
            list_max_gini[horizon].append(max(metric.compute_gini_on_run(horizon)))
    print("100/100")

    plt.figure()
    for horizon in list_horizon:
        plt.plot(list_nb_val, list_max_gini[horizon], label=horizon)
    plt.title("Maximal Relative Gini index according to the number of validators and the horizon size")
    plt.ylabel("Maximal Relative Gini index")
    plt.xlabel("Number of validators")
    plt.legend()
    plt.show()
