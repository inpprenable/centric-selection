import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def proba_select_val(n: int, p_size: int) -> float:
    k = int(n / 3) + 1
    return math.comb(n, k) / math.comb(p_size, k)


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a parcours with Genetic Algorithme')
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("--node", type=int, default=200,
                        help="Set the number of node in the graph")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_parser()

    nb_node = args.node

    plt.figure()
    arrange = np.arange(4, nb_node)
    df = pd.DataFrame(index=arrange)
    reference = np.log10(np.array([proba_select_val(n, nb_node) for n in arrange]))
    df["Reference"] = reference
    # plt.plot(arrange, reference, label="Reference")
    for alpha in np.arange(1, 2.75, 0.25):
        proba_alpha = np.log10([proba_select_val(n, min(int(alpha * n), nb_node)) for n in arrange])
        # plt.plot(arrange, np.log10(proba_alpha / reference),
        #          label=f"Alpha = {alpha:.1f}")
        df[f"Alpha = {alpha:.1f}"] = proba_alpha
        # plt.plot(arrange, proba_alpha, label=f"Alpha = {alpha:.1f}")

        proba_alpha_2 = np.log10([proba_select_val(n, min(int(alpha * (n + 20)), nb_node)) for n in arrange])
        plt.plot(arrange, proba_alpha_2, label=f"Alpha = {alpha:.1f} + 30", linestyle="--")

    plt.plot(arrange, reference, label="Reference")
    plt.xlabel("Seuil minimum de nombre de validateur")
    plt.ylabel("log10(Proba(alpha))")
    plt.title("Proba(alpha) en fonction du seuil minimum de nombre de validateur")

    plt.legend()
    if args.output is not None:
        if args.output.endswith(".csv"):
            df.to_csv(args.output, index=True, sep="\t")
        else:
            plt.savefig(args.output)
    else:
        plt.show()
