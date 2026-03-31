#!/usr/bin/env python3
import argparse
import csv
import json
import math
import multiprocessing as mp
import os.path
import time

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from sample.fonctionGraph import get_weight_graph, calcul_matrice_adjacente


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a parcours with Genetic Algorithme')
    parser.add_argument("input", type=argparse.FileType('r', encoding='utf-8'), help="The adjacent matrix file")
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("--min", type=int, default=2,
                        help="Set the minimal number of validator for the research")
    parser.add_argument("--max", type=int, default=math.inf,
                        help="Set the maximal number of validator for the research")
    parser.add_argument("--worst", help="Search for worst subgroup instead of the best", action="store_true",
                        default=False)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--pop", type=int, default=100,
                        help="Set the population size for the Genetic Algorithm")
    parser.add_argument("--gen", type=int, default=100,
                        help="Set the number of generation for the Genetic Algorithm")
    parser.add_argument("--step", type=int, default=1,
                        help="Set the step in the loop of number of validator for the research")
    args = parser.parse_args()
    return args


def read_csv(path: str) -> dict[int, dict[str, float]]:
    dico = {}
    if path is not None and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as file:
            csv_reader = csv.DictReader(file)
            line_count = 0
            for row in csv_reader:
                min_val = row["min"]
                if min_val == 0:
                    min_val = math.inf
                dico[int(row["nb_val"])] = {"min": float(min_val), "max": float(row["max"])}
                line_count += 1
    return dico


def write_csv(path: str, dico: dict) -> None:
    list_val = sorted(dico.keys())
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["nb_val", "min", "max"])
        for nb_val in list_val:
            min_val = dico[nb_val]["min"]
            if min_val == math.inf:
                min_val = 0
            writer.writerow([nb_val, min_val, dico[nb_val]["max"]])
    return


class MyFunctionalProblem(FunctionalProblem):

    def __init__(self, nb_val: int, matrix: np.ndarray, min: bool = True):
        nb_node = len(matrix)
        self.matrix = matrix
        self.nb_val = nb_val

        def func_obj(x:np.ndarray):
            return (1 if min else -1) * get_weight_graph(x, self.matrix)

        satisfy_validateur = lambda x: abs(np.sum(x) - self.nb_val)
        super().__init__(nb_node, func_obj, constr_ieq=[], constr_eq=[satisfy_validateur], xl=0, xu=1, vtype=bool)


nb_link = lambda n: int(n * (n - 1) / 2)

def create_random_solution(nb_node: int, nb_val: int) -> np.ndarray:
    solution = np.zeros(nb_node, dtype=bool)
    selected_indices = np.random.choice(nb_node, nb_val, replace=False)
    solution[selected_indices] = True
    return solution

def process_minimization(nb_val, arg_GA, matrix):
    """
    Process the minimization of the problem
    :param nb_val: the number of validator
    :param arg_GA: the arguments: [worst, pop, gen, verbose]: List[bool, int, int, int]
    :param matrix: the matrix of the graph
    :return: the number of validator and the average weight
    """
    problem = MyFunctionalProblem(nb_val, matrix, not arg_GA[0])
    algorithm = GA(pop_size=arg_GA[1],
                   sampling=BinaryRandomSampling(),
                   crossover=TwoPointCrossover(),
                   mutation=BitflipMutation(),
                   eliminate_duplicates=True)
    res = minimize(problem, algorithm, ("n_gen", arg_GA[2]), verbose=arg_GA[3] > 0,
                   save_history=False)
    weigh_graph = problem.evaluate(res.X)[0].astype(float)[0]
    avg_weight = weigh_graph / nb_link(nb_val)
    return nb_val, avg_weight


if __name__ == '__main__':
    args = create_parser()
    nb_node, matrix = None, None
    if args.input.name.endswith(".csv"):
        data = pd.read_csv(args.input, header=None)
        matrix = data.values
        nb_node = data.shape[0]
    elif args.input.name.endswith(".json"):
        data = json.loads(args.input.read())
        data = {int(k): v for k, v in data.items()}
        list_node = data
        matrix = calcul_matrice_adjacente(data)
        nb_node = len(list_node)
    else:
        raise Exception("Input file format not supported")
    args.input.close()

    min_val = args.min
    max_val = min(args.max, nb_node)

    dico = read_csv(args.output)

    arg_GA = [args.worst, args.pop, args.gen, args.verbose]


    # Force numba compilation
    nb_node = len(matrix)
    exemple = create_random_solution(nb_node, nb_node//2)
    get_weight_graph(exemple, matrix)

    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(process_minimization,
                               [(nb_val, arg_GA, matrix) for nb_val in range(min_val, max_val + 1, args.step)])
    t1 = time.time()


    for nb_val, avg_weight in results:
        if nb_val not in dico:
            dico[nb_val] = {"min": math.inf, "max": 0}
        if args.worst and dico[nb_val]["max"] < -avg_weight:
            dico[nb_val]["max"] = -avg_weight.item()
        if not args.worst and dico[nb_val]["min"] > avg_weight:
            dico[nb_val]["min"] = avg_weight.item()

        if args.output is not None:
            write_csv(args.output, dico)
        else:
            print(nb_val, dico[nb_val])
    print(f"Total time: {t1 - t0:.2f} seconds")
