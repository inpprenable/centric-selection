import argparse

import matplotlib.pyplot as plt
import torch

from sample.center_scan import gini_coefficient, gini_coefficient_worst
from sample.stake_gen import read_stake_json


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a parcours with Genetic Algorithme')
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("file", type=argparse.FileType("r"), help="The files to read")
    parser.add_argument("-k", type=int, help="The number of elements to select", default=10)
    parser.add_argument("--simulations", type=int, help="The number of simulations", default=100000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_parser()
    stake = read_stake_json(args.file)
    stake = torch.Tensor(stake)

    N = stake.size(0)
    selected_count = torch.zeros(N)  # Compte combien de fois chaque élément est sélectionné

    evol_gini = torch.zeros(args.simulations)

    # Simulation Monte Carlo
    for i in range(args.simulations):
        remaining_stakes = stake.clone()  # Copie des stakes pour ne pas modifier les originaux
        selected = torch.multinomial(remaining_stakes, args.k, replacement=False)  # Tirer k éléments sans remise
        selected_count[selected] += 1

        # Calcul de la probabilité
        # probabilities = selected_count / i
        # evol_gini[i] = gini_coefficient(probabilities.numpy())
    last_gini = gini_coefficient(selected_count.numpy() / args.simulations)
    print(f"Gini coefficient : {last_gini}")
    print(f"Gini R : {last_gini / gini_coefficient_worst(args.k, N)}")

    # delta_gini = (evol_gini - last_gini).abs()/last_gini * 100
    #
    # plt.figure()
    # plt.plot(delta_gini)
    # plt.show()
