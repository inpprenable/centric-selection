import argparse
import math
import time

import torch

from sample.center_scan import gini_coefficient, worst_standard_deviation, \
    gini_coefficient_worst
from sample.center_scan import write_csv, read_csv
from sample.center_stake import ConfigSelection, Frame, stake_distribution
from sample.center_torch import Matrix, Metric
from sample.stake_gen import read_stake_json


def is_correct_float(value: str) -> float:
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid float")
    if not (value >= 1):
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid float, it must be greater than 1")
    return value


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a range of parcours with our selection')
    parser.add_argument("input", type=str, help="The position node file")
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("--mu", type=float, help="The coefficient in the calculus", default=1)
    parser.add_argument("--min", type=int, default=4,
                        help="Set the minimal number of validator for the research")
    parser.add_argument("--elipse", type=int, default=0,
                        help="Set the elipse of iteration before the sampling")
    parser.add_argument("--max", type=int, default=math.inf,
                        help="Set the maximal number of validator for the research")
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--conf", help="Print confidence intervals", action="store_true", default=False)
    parser.add_argument("--gen", type=int, default=1,
                        help="Set the number of generation")
    parser.add_argument("--step", type=int, default=1,
                        help="Set the step in the loop of number of validator for the research")
    parser.add_argument("--horizon", type=int, default=200,
                        help="Set the horizon of the simulation")
    parser.add_argument("-r", "--random", type=is_correct_float, default=1,
                        help="The random factor, selected nodes are chosen from the random factor * N closest nodes")
    parser.add_argument("-s", "--stake", type=argparse.FileType("r"), default=None,
                        help="The stake distribution file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_parser()
    liste_node_matrix = Matrix.create_from_file(args.input)

    nb_node = len(liste_node_matrix.init_list_node)
    min_val = args.min
    max_val = min(args.max, nb_node)
    nb_gen = args.gen
    elipse = args.elipse

    if args.stake is not None:
        stake = read_stake_json(args.stake)
    else:
        stake = stake_distribution(nb_node)

    print("Gini stake : ", gini_coefficient(stake))
    stake = torch.Tensor(stake)

    neutral_stake = torch.ones(nb_node)

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
            config = ConfigSelection(args.mu, args.random)
            frame = Frame(nb_node, args.horizon, config)
            metric = Metric(nb_node, 0, False)

            for i in range(elipse):
                # liste_node_matrix.update()
                frame.update(liste_node_matrix.current_matrix, nb_val, neutral_stake)

            t1 = time.time()
            for i in range(nb_gen):
                frame.update(liste_node_matrix.current_matrix, nb_val, stake)
                metric.update_frame(frame, liste_node_matrix.current_matrix)
            t2 = time.time()
            avg_weight_mean, avg_weight_std = metric.get_avg_weight_info()
            time_validator = metric.get_time_validator()
            gini_index = gini_coefficient(time_validator)
            dico[nb_val] = {"avg_weight": avg_weight_mean, "nb_gen": nb_gen,
                            "std": avg_weight_std,
                            "std_r": 0 if nb_val == nb_node else avg_weight_std / worst_standard_deviation(nb_val,
                                                                                                           nb_node),
                            "time_val_avg": time_validator.mean() / nb_gen,
                            "time_val_std": time_validator.std() / nb_gen,
                            "temps_calcul": (t2 - t1) / nb_gen,
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
