import json
import os
import time
from abc import ABC

import numpy as np
import torch

from sample.center_scan import create_parser, gini_coefficient, worst_standard_deviation, \
    gini_coefficient_worst, write_csv
from sample.fonctionGraph import calcul_matrice_adjacente, generate_node


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
    def __init__(self, coef_eloignement: float):
        self.coef_eloignement = coef_eloignement

    def func_eloignement(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(t / self.coef_eloignement)


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
        # Sélectionner les 'nb_val' premiers indices (les plus proches)
        return sorted_indices[:nb_val]

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
    def __init__(self, nb_node: int):
        self.nb_node = nb_node
        self.avg_weight, self.median_weight, self.max_weight = [], [], []
        self.timestamp = 0
        self.time_validator = torch.zeros(nb_node, dtype=torch.float)
        self.prev_val = None

    def update_frame(self, frame: Frame, matrix: torch.Tensor):
        self.prev_val = frame.validators.clone()
        self.timestamp += 1
        weights = frame.calcul_metric(matrix)
        self.avg_weight.append(weights[0])
        # self.median_weight.append(weights[1])
        # self.max_weight.append(weights[2])

        increment = torch.zeros_like(self.time_validator)
        increment[frame.validators] = 1
        self.time_validator += increment

    def get_avg_weight_info(self) -> tuple:
        """
        Return the mean and the standard deviation of the average
        """
        if len(self.avg_weight) == 0:
            return 0, 0
        avg_weight = np.array(self.avg_weight)
        return np.mean(avg_weight), np.std(avg_weight)


if __name__ == '__main__':
    args = create_parser()
    liste_node_matrix = Matrix.create_from_file(args.input)

    nb_node = len(liste_node_matrix.init_list_node)
    min_val = args.min
    max_val = min(args.max, nb_node)
    nb_gen = args.gen
    elipse = args.elipse
    # dico = read_csv(args.output)
    dico = {}
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
            gini_index = gini_coefficient(metric.time_validator.numpy())
            dico[nb_val] = {"avg_weight": avg_weight_mean, "nb_gen": nb_gen,
                            "std": avg_weight_std,
                            "std_r": 0 if nb_val == nb_node else avg_weight_std / worst_standard_deviation(nb_val,
                                                                                                           nb_node),
                            "time_val_avg": metric.time_validator.mean().item() / nb_gen,
                            "time_val_std": metric.time_validator.std().item() / nb_gen,
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
