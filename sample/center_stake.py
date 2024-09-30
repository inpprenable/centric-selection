import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sample.center_scan import gini_coefficient, worst_standard_deviation, \
    gini_coefficient_worst
from sample.center_torch import Metric, PastValidatorUnbounded, PastValidatorWindow, Matrix
from sample.stake_gen import stake_distribution, read_stake_json


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
    parser.add_argument("--nb", type=int, default=10,
                        help="Set the default number of validator for the research")
    parser.add_argument("--elipse", type=int, default=0,
                        help="Set the elipse of iteration before the sampling")
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--conf", help="Print confidence intervals", action="store_true", default=False)
    parser.add_argument("--gen", type=int, default=1,
                        help="Set the number of generation")
    parser.add_argument("--horizon", type=int, default=200,
                        help="Set the horizon of the simulation")
    parser.add_argument("-r", "--random", type=is_correct_float, default=1,
                        help="The random factor, selected nodes are chosen from the random factor * N closest nodes")
    parser.add_argument("-s", "--stake", type=argparse.FileType("r"), default=None,
                        help="The stake distribution file")
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

    def update(self, matrix: torch.Tensor, nb_val: int, stake: torch.Tensor):
        self.timestamp += 1
        self.nb_val = nb_val
        self.center = select_center(self.validators, matrix)
        self.validators = self.select_new_val(self.center, nb_val, matrix)
        self.update_past(stake)

    def update_past(self, stake: torch.Tensor):
        N = self.nb_node
        validator_bool = torch.zeros(N, dtype=torch.bool)
        validator_bool[self.validators] = True

        power = 0.5 + (self.nb_val / N) ** 2 * 1.5
        pow_state = stake.float().pow(power)

        adjustement = torch.zeros(N, dtype=torch.float)
        if self.nb_val != self.nb_node:
            nv_stake = pow_state[~validator_bool]
            adjustement[~validator_bool] = -nv_stake / nv_stake.sum()

            v_stake = 1 / (pow_state[validator_bool])
            adjustement[validator_bool] = v_stake / v_stake.sum()

        else:
            adjustement = torch.ones(N, dtype=torch.float) / N
        self.past_validator.update(adjustement)

    def calcul_metric(self, matrix: torch.Tensor) -> tuple:
        list_weight = get_list_weight_torch(self.validators, matrix)
        return list_weight.mean(), list_weight.median(), list_weight.max()


def monte_carlo_selection(stake: torch.Tensor, k: int, num_simulations: int) -> torch.Tensor:
    N = stake.size(0)
    selected_count = torch.zeros(N).to(stake.device)  # Compte combien de fois chaque élément est sélectionné

    # Simulation Monte Carlo
    for _ in range(num_simulations):
        remaining_stakes = stake.clone()  # Copie des stakes pour ne pas modifier les originaux
        selected = torch.multinomial(remaining_stakes, k, replacement=False)  # Tirer k éléments sans remise
        selected_count[selected] += 1

    # Calcul de la probabilité
    probabilities = selected_count / num_simulations
    return probabilities


if __name__ == '__main__':
    args = create_parser()
    liste_node_matrix = Matrix.create_from_file(args.input)

    nb_node = len(liste_node_matrix.init_list_node)
    nb_gen = args.gen
    elipse = args.elipse

    nb_val = args.nb

    config = ConfigSelection(args.mu, args.random)
    frame = Frame(nb_node, args.horizon, config)
    metric = Metric(nb_node, 0, True)

    if args.stake is not None:
        stake = read_stake_json(args.stake)
    else:
        stake = stake_distribution(nb_node)

    print("Gini stake : ", gini_coefficient(stake))
    stake = torch.Tensor(stake)

    neutral_stake = torch.ones(nb_node)
    # stake = neutral_stake
    torch.no_grad()

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

    proba = monte_carlo_selection(stake, nb_val, 10000)

    if args.output is None:
        metrics = {"avg_weight": avg_weight_mean, "nb_gen": nb_gen,
                   "std": avg_weight_std,
                   "std_r": 0 if nb_val == nb_node else avg_weight_std / worst_standard_deviation(nb_val,
                                                                                                  nb_node),
                   "time_val_avg": time_validator.mean() / nb_gen,
                   "time_val_std": time_validator.std() / nb_gen,
                   "temps_calcul": (t2 - t1) / nb_gen,
                   "gini_coef": gini_index,
                   "gini_coef_r": 0 if nb_val == nb_node else gini_index / gini_coefficient_worst(nb_val,
                                                                                                  nb_node), }
        print(metrics)

        plt.figure()
        # Première sous-figure (Gini index)
        plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, 1ère position
        plt.plot(metric.gini)
        plt.title("Gini Index")

        # Deuxième sous-figure (Average weight)
        plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, 2ème position
        plt.plot(metric.avg_weight)
        plt.title("Average Weight")

        # Afficher la figure avec les deux sous-figures
        plt.tight_layout()  # Pour éviter que les titres et axes se chevauchent

        plt.show()

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(stake / stake.sum(), label=f'Gaussian Distribution (mean={75}, std_dev={10})', drawstyle='steps-post')
        plt.title('Stake Distribution')
        plt.subplot(3, 1, 2)
        plt.plot(time_validator / nb_gen, label='Time Validator')
        plt.plot(proba, label='Time Validator', linestyle='--')
        plt.title('Time Validator')
        plt.subplot(3, 1, 3)
        sum_val = frame.get_sum_validators()
        plt.plot(sum_val, label='Sum Validators')
        plt.title(f'Sum Validators : {sum_val.sum()}')
        plt.show()
    else:
        df = pd.DataFrame(
            {"id_node": range(1, nb_node + 1), "time": time_validator / nb_gen,
             "proba_select": proba, }
        )
        df.set_index("id_node", inplace=True)
        df.to_csv(args.output, index=True, sep="\t")
