import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from sample.center_scan import gini_coefficient


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a stake distribution')
    parser.add_argument("-n", "--nb_node", type=int, default=200, )
    parser.add_argument("-i", "--input", type=argparse.FileType("r"),
                        help="the file to read the stake distribution", default=None)
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("-r", "--random", action="store_true", default=False,
                        help="Generate a random stake distribution")
    args = parser.parse_args()
    return args


def stake_distribution(nb_node: int) -> np.ndarray:
    # Définir les paramètres de la distribution gaussienne
    mean = 75  # Moyenne de la distribution
    std_dev = 10  # Écart-type

    # Créer les données pour l'abscisse (de 0 à 200) avec des valeurs entières
    x = np.arange(nb_node)
    # Calculer la distribution gaussienne pour chaque valeur de x
    y = norm.pdf(x, mean, std_dev)

    y += norm.pdf(x, mean + 50, std_dev) * 0.5

    y = y * 5

    # Arrondir les valeurs de y à l'entier supérieur et s'assurer qu'elles sont >= 1
    y = np.ceil(y * 250).astype(int)
    y[y < 1] = 1
    y = y + 1
    return y


def random_stake_distribution(nb_node: int) -> np.ndarray:
    stake = np.random.randint(10, 15, nb_node)
    stake.sort()
    # stake = stake * 1000 / sum
    return stake.astype(int)


def write_stake_json(path: str, stake: np.ndarray) -> None:
    # Créer un dictionnaire avec les données de stake
    stake_dict = {"stake": stake.tolist(),
                  "Gini": gini_coefficient(stake),
                  "Sum": np.sum(stake)}
    # Écrire le dictionnaire dans un fichier JSON
    with open(path, "w") as file:
        json.dump(stake_dict, file, default=str)


def read_stake_json(file):
    stake = json.load(file)
    file.close()
    return np.array(stake["stake"])


if __name__ == '__main__':
    args = create_parser()

    if args.input is not None:
        stake = read_stake_json(args.input)
    else:
        if args.random:
            stake = random_stake_distribution(args.nb_node)
        else:
            stake = stake_distribution(args.nb_node)
        # stake = np.ones(args.nb_node, dtype=int)
    if args.output is not None:
        write_stake_json(args.output, stake)
    else:
        print("Gini stake : ", gini_coefficient(stake))
        print("Sum of stake : ", np.sum(stake))
        plt.figure()
        plt.plot(stake)
        plt.xlabel("Validator index")
        plt.ylabel("Stake amount")
        plt.ylim(0)
        plt.show()
