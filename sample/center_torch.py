import argparse
import json
import math
import os
import time
from abc import ABC

import numpy as np
import torch

from sample.center_scan import gini_coefficient, worst_standard_deviation, \
    gini_coefficient_worst, write_csv, read_csv
from sample.fonctionGraph import calcul_matrice_adjacente, generate_node


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
    parser.add_argument("--horizon", type=int, default=200,
                        help="Set the horizon of the simulation")
    parser.add_argument("-r", "--random", type=is_correct_float, default=1,
                        help="The random factor, selected nodes are chosen from the random factor * N closest nodes")
    args = parser.parse_args()
    return args


def select_center(list_validators: torch.Tensor, matrix: torch.Tensor) -> int:
    N = matrix.size(0)
    # list_average_delay = [sum(delays) for delays in matrix]
    list_average_delay = torch.sum(matrix[:, list_validators], dim=1)
    # list_average_delay = [sum(delays[list_validators]) for delays in matrix]
    # Créer la liste des centres potentiels à partir des validateurs
    list_potential_center = list_validators.clone()

    # Si `list_potential_center` est vide, créer une liste avec tous les indices qui ne sont pas dans `list_validators`
    if list_potential_center.numel() == 0:
        list_potential_center = torch.tensor([node for node in range(N) if node not in list_validators])

    # Filtrer les `list_average_delay` pour ne conserver que les valeurs correspondant aux centres potentiels
    potential_delays = list_average_delay[list_potential_center]

    # Trouver l'indice du centre avec le délai minimum
    min_idx = torch.argmin(potential_delays)

    # Récupérer l'indice original du centre correspondant dans `list_potential_center`
    center = list_potential_center[min_idx].item()
    return center


def get_list_weight_torch(validators: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    N = len(validators)

    # Créer un masque pour obtenir les indices i, j tels que i < j
    indices_i, indices_j = torch.triu_indices(N, N, offset=1)

    # Sélectionner les nœuds correspondants dans validators
    nodes_i = validators[indices_i]
    nodes_j = validators[indices_j]

    # Récupérer les poids correspondants dans la matrice
    list_poid = matrix[nodes_i, nodes_j]

    return list_poid.floor()


class ConfigSelection:
    def __init__(self, coef_eloignement: float, random_factor: float == 1):
        self.coef_eloignement = coef_eloignement
        self.random_factor = random_factor

    def func_eloignement(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(t / self.coef_eloignement)

    def get_random_factor(self, sorted_indices: torch.Tensor, nb_val: int) -> torch.Tensor:
        # Si le facteur aléatoire est 1, retourner les nb_val premiers éléments
        if self.random_factor == 1:
            return sorted_indices[:nb_val]

        N = sorted_indices.size(0)

        # Calculer le nombre maximum d'indices à considérer, en limitant à N
        max_indices_to_select = min(int(self.random_factor * nb_val), N)

        # Générer une permutation aléatoire des indices dans cet intervalle
        indices = torch.randperm(max_indices_to_select)[:nb_val]

        # Retourner les éléments correspondants dans sorted_indices
        return sorted_indices[indices]


class PastValidator(ABC):
    def __init__(self, nb_node: int):
        self.nb_node = nb_node

    def get_sum_validators(self) -> torch.Tensor:
        pass

    def update(self, adjustment: torch.Tensor):
        pass


class PastValidatorWindow(PastValidator):
    def __init__(self, horizon: int, nb_node: int):
        super().__init__(nb_node)
        self.horizon = horizon
        self._current_view = 0
        self.past_validator = torch.zeros((horizon, nb_node), dtype=torch.float)

    def get_sum_validators(self) -> torch.Tensor:
        return self.past_validator.sum(dim=0)

    def update(self, adjustment: torch.Tensor):
        self.past_validator[self._current_view] = adjustment
        self._current_view = (self._current_view + 1) % self.horizon


class PastValidatorUnbounded(PastValidator):
    def __init__(self, nb_node: int):
        super().__init__(nb_node)
        self.past_validator = torch.zeros(nb_node, dtype=torch.float)

    def get_sum_validators(self) -> torch.Tensor:
        return self.past_validator

    def update(self, adjustment: torch.Tensor):
        self.past_validator += adjustment


class Frame:
    def __init__(self, nb_node: int, horizon: int, config: ConfigSelection):
        self.timestamp = 0
        self.nb_node = nb_node
        self.nb_val = nb_node
        self.config = config
        self._current_view = 0
        if horizon == 0:
            self.past_validator = PastValidatorUnbounded(nb_node)
        else:
            self.past_validator = PastValidatorWindow(horizon, nb_node)
        self.validators = torch.arange(nb_node)

    def get_sum_validators(self) -> torch.Tensor:
        return self.past_validator.get_sum_validators()

    def select_new_val(self, center: int, nb_val: int, matrix: torch.Tensor) -> torch.Tensor:
        delays = matrix[center, :] * self.config.func_eloignement(self.get_sum_validators())
        # Obtenir les indices triés en fonction des délais
        _, sorted_indices = torch.sort(delays)
        # # Sélectionner les 'nb_val' premiers indices (les plus proches)
        # return sorted_indices[:nb_val]
        return self.config.get_random_factor(sorted_indices, nb_val)

    def update(self, matrix: torch.Tensor, nb_val: int):
        self.timestamp += 1
        self.nb_val = nb_val
        self.center = select_center(self.validators, matrix)
        self.validators = self.select_new_val(self.center, nb_val, matrix)
        self.uptade_past()

    def uptade_past(self):
        N = self.nb_node
        if self.nb_val != self.nb_node:
            adjustement = -torch.ones(N, dtype=torch.float) / (N - self.nb_val)
            adjustement[self.validators] = 1 / self.nb_val
        else:
            adjustement = torch.ones(N, dtype=torch.float) / N
        self.past_validator.update(adjustement)

    def calcul_metric(self, matrix: torch.Tensor) -> tuple:
        list_weight = get_list_weight_torch(self.validators, matrix)
        return list_weight.mean(), list_weight.median(), list_weight.max()


class Matrix:
    def __init__(self, height: float, width: float, nb_node: int):
        self.init_list_node = generate_node(height, width, nb_node)
        self.current_list_node = self.init_list_node.copy()
        init_matrix = calcul_matrice_adjacente(self.init_list_node)
        self.init_matrix = torch.tensor(init_matrix, dtype=torch.float)
        self.current_matrix = self.init_matrix
        self.counter = 0

    @staticmethod
    def create_from_file(filename: str) -> 'Matrix':
        if filename == "" or not os.path.exists(filename):
            raise Exception("The file doesn't exist")
        with open(filename, "r") as file:
            list_node = json.loads(file.read())
            list_node = {int(k): v for k, v in list_node.items()}
        list_node_mat = Matrix(100, 100, 100)
        list_node_mat.init_list_node = list_node
        list_node_mat.current_list_node = list_node_mat.init_list_node.copy()
        init_matrix = calcul_matrice_adjacente(list_node_mat.init_list_node)
        list_node_mat.init_matrix = torch.tensor(init_matrix, dtype=torch.float)
        list_node_mat.current_matrix = list_node_mat.init_matrix
        return list_node_mat


class Metric:
    def __init__(self, nb_node: int, horizon: int = 0, follow_gini: bool = False):
        self.nb_node = nb_node
        self.avg_weight = []
        self.timestamp = 0
        self._follow_gini = follow_gini
        self.gini = []
        if horizon == 0:
            self.past_validator = PastValidatorUnbounded(nb_node)
        else:
            self.past_validator = PastValidatorWindow(horizon, nb_node)

    def update_frame(self, frame: Frame, matrix: torch.Tensor):
        self.timestamp += 1
        weights = frame.calcul_metric(matrix)
        self.avg_weight.append(weights[0])

        increment = torch.zeros(self.nb_node, dtype=torch.float)
        increment[frame.validators] = 1
        self.past_validator.update(increment)
        if self._follow_gini:
            self.gini.append(gini_coefficient(self.past_validator.get_sum_validators().numpy()))

    def get_avg_weight_info(self) -> tuple:
        """
        Return the mean and the standard deviation of the average
        """
        if len(self.avg_weight) == 0:
            return 0, 0
        avg_weight = np.array(self.avg_weight)
        return np.mean(avg_weight), np.std(avg_weight)

    def get_time_validator(self) -> np.ndarray:
        return self.past_validator.get_sum_validators().numpy()


if __name__ == '__main__':
    args = create_parser()
    liste_node_matrix = Matrix.create_from_file(args.input)

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
            config = ConfigSelection(args.mu, args.random)
            frame = Frame(nb_node, args.horizon, config)
            metric = Metric(nb_node)

            for i in range(elipse):
                # liste_node_matrix.update()
                frame.update(liste_node_matrix.current_matrix, nb_val)

            t1 = time.time()
            for i in range(nb_gen):
                frame.update(liste_node_matrix.current_matrix, nb_val)
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
