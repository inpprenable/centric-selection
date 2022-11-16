#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import numpy as np

from fonctionGraph import calcul_matrice_adjacente, get_weight_graph, list_int_to_list_bool


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a range of parcours with random selection')
    parser.add_argument("input", type=argparse.FileType('r', encoding='utf-8'), help="The adjacent matrix file")
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("--min", type=int, default=4,
                        help="Set the minimal number of validator for the research")
    parser.add_argument("--max", type=int, default=math.inf,
                        help="Set the maximal number of validator for the research")
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--conf", help="Print confidence intervals", action="store_true", default=False)
    parser.add_argument("--gen", type=int, default=1,
                        help="Set the number of generation")
    args = parser.parse_args()
    return args


def read_csv(path: str) -> dict:
    dico = {}
    if path is not None and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                dico[int(row["nb_val"])] = {"avg_weight": float(row["avg_weight"]), "nb_gen": int(row["nb_gen"]),
                                            "std": float(row["std"])}
    return dico


def write_csv(path: str, dico: dict) -> None:
    list_val = sorted(dico.keys())
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["nb_val", "avg_weight", "nb_gen", "std"])
        for nb_val in list_val:
            writer.writerow([nb_val, dico[nb_val]["avg_weight"], dico[nb_val]["nb_gen"], dico[nb_val]["std"]])
    return


def get_random_weight(nb_val: int, matrix: np.ndarray, gen: int) -> np.ndarray:
    assert gen > 0
    assert 3 < nb_val <= len(matrix)
    history = np.zeros(gen, dtype=int)
    list_nodes = np.arange(len(matrix))
    for i in range(gen):
        validators = np.random.choice(list_nodes, size=nb_val, replace=False)
        history[i] = get_weight_graph(list_int_to_list_bool(len(matrix), validators), matrix)
    return history


nb_link_f = lambda n: int(n * (n - 1) / 2)

if __name__ == '__main__':
    args = create_parser()
    data = json.loads(args.input.read())
    list_node, matrix = None, None
    if type(data) == list:
        matrix = np.array(data)
    elif type(data) == dict:
        data = {int(k): v for k, v in data.items()}
        list_node = data
        matrix = calcul_matrice_adjacente(data)
    args.input.close()

    nb_node = len(list_node)
    min_val = args.min
    max_val = min(args.max, nb_node)
    nb_gen = args.gen
    dico = read_csv(args.output)

    total = max_val + 1 - min_val
    for nb_val in range(min_val, max_val + 1):
        i = nb_val - min_val
        if total > 10 and i % (total // 10) == 0:
            print("{}/100".format(int(100 * i / total)))

        if nb_val not in dico:
            dico[nb_val] = {"avg_weight": 0, "nb_gen": 0, "std": 0}
        if nb_gen > dico[nb_val]["nb_gen"]:
            list_weight = get_random_weight(nb_val, matrix, args.gen)
            nb_link = nb_link_f(nb_val)
            dico[nb_val] = {"avg_weight": list_weight.mean() / nb_link, "nb_gen": nb_gen,
                            "std": list_weight.std() / nb_link}

            if args.output is not None:
                write_csv(args.output, dico)
            else:
                print("The average random set of {} validators has an average weight of {:.2f}, with an STD of {:.2f}."
                      .format(nb_val, list_weight.mean() / nb_link, list_weight.std() / nb_link))
