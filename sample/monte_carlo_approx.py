import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sample.center_stake import proba_select, monte_carlo_selection
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

    index = np.arange(1, stake.size(0) + 1)
    monte_carlo = monte_carlo_selection(stake, args.k, args.simulations)
    computation_est = proba_select(stake, args.k)

    print(f"Sum Monte Carlo : {monte_carlo.sum()}")
    print(f"Sum Computation : {computation_est.sum()}")

    if args.output is None:
        plt.figure()
        plt.plot(index, monte_carlo, label="Monte Carlo")
        plt.plot(index, computation_est, label="Computation")
        plt.legend()
        plt.show()
    else:
        if os.path.exists(args.output):
            df = pd.read_csv(args.output, sep="\t")
        else:
            df = pd.DataFrame({"id_node": index})
        df[f"Monte Carlo"] = monte_carlo.numpy()
        df.set_index("id_node", inplace=True)
        df.to_csv(args.output, index=True, sep="\t")
