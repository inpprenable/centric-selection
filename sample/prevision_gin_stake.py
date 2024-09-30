import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from center_scan import gini_coefficient, gini_coefficient_worst
from sample.center_stake import monte_carlo_selection
from sample.stake_gen import read_stake_json


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a parcours with Genetic Algorithme')
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("files", type=argparse.FileType("r"), nargs='+', help="The files to read")
    parser.add_argument("--simulations", type=int, help="The number of simulations", default=100000)
    args = parser.parse_args()
    return args


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

if __name__ == '__main__':
    args = create_parser()
    df = None
    plt.figure()
    t1 = time.time()
    for file in args.files:
        label = file.name.split('/')[-1].split('.json')[0]
        print("Processing ", label)
        stake = read_stake_json(file)
        stake = torch.Tensor(stake)
        stake = stake.to(device)
        nb_node = len(stake)
        index = range(4, nb_node + 1)

        predicted_gini_r = np.zeros(nb_node - 3)
        gini_w = np.zeros(nb_node - 3)
        for i in index:
            proba = monte_carlo_selection(stake, i, args.simulations)
            predicted_gini = gini_coefficient(proba.cpu().numpy())

            gini_w = gini_coefficient_worst(i, nb_node)
            predicted_gini_r[i] = predicted_gini / (gini_w if i != nb_node else 1)

        df_temp = pd.DataFrame({label: predicted_gini_r, "nb_val": index})
        if df is None:
            df = df_temp
        else:
            df = pd.merge(df, df_temp, on="nb_val", how="outer")
        plt.plot(index, predicted_gini_r, label=label)
    df.set_index("nb_val", inplace=True)
    t2 = time.time()
    print("Time : ", t2 - t1)
    if args.output is not None:
        if args.output.endswith(".csv"):
            df.to_csv(args.output, index=True, sep="\t")
        else:
            plt.savefig(args.output)
    else:
        plt.show()
