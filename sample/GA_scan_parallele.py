#!/usr/bin/env python3
import argparse
import csv
import json
import math
import multiprocessing as mp
import os.path

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from fonctionGraph import calcul_matrice_adjacente, get_weight_graph


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
    args = parser.parse_args()
    return args


def read_csv(path: str) -> dict:
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

        def func_obj(x: list[bool]):
            return (1 if min else -1) * get_weight_graph(x, self.matrix)

        satisfy_validateur = lambda x: abs(np.sum(x) - self.nb_val)
        super().__init__(nb_node, func_obj, constr_ieq=[], constr_eq=[satisfy_validateur], xl=0, xu=1, vtype=bool)


nb_link = lambda n: int(n * (n - 1) / 2)


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
                   save_history=True)
    weigh_graph = problem.evaluate(res.X)[0].astype(float)[0]
    avg_weight = weigh_graph / nb_link(nb_val)

    return nb_val, avg_weight


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

    dico = read_csv(args.output)

    arg_GA = [args.worst, args.pop, args.gen, args.verbose]

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(process_minimization,
                               [(nb_val, arg_GA, matrix) for nb_val in range(min_val, max_val + 1)])

    for nb_val, avg_weight in results:
        if nb_val not in dico:
            dico[nb_val] = {"min": math.inf, "max": 0}
        if args.worst and dico[nb_val]["max"] < -avg_weight:
            dico[nb_val]["max"] = -avg_weight
        if not args.worst and dico[nb_val]["min"] > avg_weight:
            dico[nb_val]["min"] = avg_weight

        if args.output is not None:
            write_csv(args.output, dico)
        else:
            print(nb_val, dico[nb_val])
