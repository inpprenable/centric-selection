#!/usr/bin/env python3

import argparse
import csv
import math
import os
import time

import numpy as np

from center_graphical import ConfigSelection, Frame, Matrix


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a range of parcours with our selection')
    parser.add_argument("input", type=str, help="The position node file")
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("--mu", type=float, help="The coefficient in the calculus", default=1)
    parser.add_argument("--min", type=int, default=4,
                        help="Set the minimal number of validator for the research")
    parser.add_argument("--elipse", type=int, default=10,
                        help="Set the elipse of iteration before the sampling")
    parser.add_argument("--max", type=int, default=math.inf,
                        help="Set the maximal number of validator for the research")
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--conf", help="Print confidence intervals", action="store_true", default=False)
    parser.add_argument("--gen", type=int, default=1,
                        help="Set the number of generation")
    parser.add_argument("--step", type=int, default=1,
                        help="Set the step in the loop of number of validator for the research")
    args = parser.parse_args()
    return args


def read_csv(path: str) -> dict:
    dico = {}
    if path is not None and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                dico[int(row["nb_val"])] = {"avg_weight": float(row["avg_weight"]), "nb_gen": int(row["nb_gen"]),
                                            "std": float(row["std"]),
                                            "std_r": float(row["std_r"]),
                                            "time_val_avg": float(row["time_val_avg"]),
                                            "time_val_std": float(row["time_val_std"]),
                                            "temps_calcul": float(row["temps_calcul"]),
                                            "gini_coef": float(row["gini_coef"]),
                                            "gini_coef_r": float(row["gini_coef_r"])}
    return dico


def write_csv(path: str, dico: dict) -> None:
    list_val = sorted(dico.keys())
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["nb_val", "avg_weight", "nb_gen", "std", "std_r", "time_val_avg", "time_val_std", "temps_calcul",
             "gini_coef", "gini_coef_r"])
        for nb_val in list_val:
            writer.writerow(
                [nb_val, dico[nb_val]["avg_weight"], dico[nb_val]["nb_gen"], dico[nb_val]["std"], dico[nb_val]["std_r"],
                 dico[nb_val]["time_val_avg"], dico[nb_val]["time_val_std"], dico[nb_val]["temps_calcul"],
                 dico[nb_val]["gini_coef"], dico[nb_val]["gini_coef_r"]])
    return


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


def gini_coefficient_worst(nb_val: int, nb_node: int) -> float:
    """Return the worst Gini index obtains with this parameters"""
    if nb_val == nb_node:
        return 0
    return 1 - nb_val / nb_node


def worst_standard_deviation(nb_val: int, nb_node: int) -> float:
    """Return the worst standard deviation obtains with this parameters"""
    if nb_val == nb_node:
        return 0
    return 1 / 2 * np.sqrt(1 - np.power(2 * nb_val / nb_node - 1, 2))


class Metric:
    def __init__(self, nb_node: int):
        self.nb_node = nb_node
        self.avg_weight, self.median_weight, self.max_weight = [], [], []
        self.timestamp = 0
        self.time_validator = np.zeros(nb_node, dtype=int)
        self.prev_val = None

    def update_frame(self, frame: Frame, matrix: np.ndarray):
        self.prev_val = frame.validators.copy()
        self.timestamp += 1
        weights = frame.calcul_metric(matrix)
        self.avg_weight.append(weights[0])
        self.median_weight.append(weights[1])
        self.max_weight.append(weights[2])
        for node in frame.validators:
            self.time_validator[node] += 1


if __name__ == '__main__':
    args = create_parser()
    liste_node_matrix = Matrix.create_from_file(args.input, 0, 0)

    nb_node = len(liste_node_matrix.init_list_node)
    min_val = args.min
    max_val = min(args.max, nb_node)
    nb_gen = args.gen
    elipse = args.elipse
    dico = read_csv(args.output)

    total = max_val + 1 - min_val
    for nb_val in range(min_val, max_val + 1, args.step):
        i = nb_val - min_val
        if total > 10 and i % (total // 10) == 0:
            print("{}/100".format(int(100 * i / total)))

        if nb_val not in dico:
            dico[nb_val] = {"avg_weight": 0, "nb_gen": 0, "std": 0, "std_r": 0,
                            "time_val_avg": 0, "time_val_std": 0, "temps_calcul": 0,
                            "gini_coef": 0, "gini_coef_r": 0}
        if nb_gen > dico[nb_val]["nb_gen"]:
            config = ConfigSelection(args.mu)
            frame = Frame(nb_node, config)
            metric = Metric(nb_node)

            for i in range(elipse):
                # liste_node_matrix.update()
                frame.update(liste_node_matrix.current_matrix, nb_val)

            t1 = time.time()
            for i in range(nb_gen):
                frame.update(liste_node_matrix.current_matrix, nb_val)
                metric.update_frame(frame, liste_node_matrix.current_matrix)
            t2 = time.time()
            avg_weight = np.array(metric.avg_weight)
            time_val = np.array(metric.time_validator)
            std = avg_weight.std()
            gini_index = gini_coefficient(metric.time_validator)
            dico[nb_val] = {"avg_weight": avg_weight.mean(), "nb_gen": nb_gen,
                            "std": std,
                            "std_r": 0 if nb_val == nb_node else std / worst_standard_deviation(nb_val, nb_node),
                            "time_val_avg": time_val.mean() / nb_gen,
                            "time_val_std": time_val.std() / nb_gen, "temps_calcul": (t2 - t1) / nb_gen,
                            "gini_coef": gini_index,
                            "gini_coef_r": 0 if nb_val == nb_node else gini_index / gini_coefficient_worst(nb_val,
                                                                                                           nb_node), }

            if args.output is not None:
                write_csv(args.output, dico)
            else:
                print(
                    "The average random set of {} validators has an average weight of {:.2f}, with an STD of {:.2f} in {} iteration."
                    .format(nb_val, dico[nb_val]["avg_weight"], dico[nb_val]["std"], nb_gen))
                print("The relative STD is equal to {:.3f}".format(dico[nb_val]["std_r"]))
                print("It took an average {:.4f}ms for an iteration".format(1e3 * dico[nb_val]["temps_calcul"]))
                print("The average percent of time validator is {:.2f}, with an STD of {:.4f}, ideal : {}".format(
                    dico[nb_val]["time_val_avg"], dico[nb_val]["time_val_std"], nb_val / nb_node))
                print("The Gini index is equal to {:.3f}".format(dico[nb_val]["gini_coef"]))
                print("The relative Gini Index is equal to {:.3f}".format(dico[nb_val]["gini_coef_r"]))
