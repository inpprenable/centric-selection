import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a parcours with Genetic Algorithme')
    # parser.add_argument("input", type=str, help="The gain file")
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("--node", type=int, default=200,
                        help="Set the number of node in the graph")
    parser.add_argument("--csv", type=str, default=None, help="Export data in a csv format")
    args = parser.parse_args()
    return args


def proba_select_val(m: int, n: int, p_size: int) -> float:
    k = int(n / 3) + 1
    if k > m and p_size - int(2 * n / 3) <= m:
        return 0

    if k > m:
        return 0
    if p_size - int(2 * n / 3) <= m:
        return 1
    sum = 0
    for i in range(k, n + 1):
        sum += math.comb(m, i) * math.comb(p_size - m, n - i) / math.comb(p_size, n)
    return sum


if __name__ == '__main__':
    args = create_parser()

    # df = pd.read_csv(args.input)

    # at peak, n = 4, n2 = 4 +27
    # plateau, n2 = n + 19 -> n =  12
    n, N, alpha = 12, 200, 1.5
    n2 = n + 19

    x = []
    y_rd, y_alpha = [], []

    for k in range(4, N + 1):
        proba_A = proba_select_val(k, n, N)

        proba_B = proba_select_val(k, n2, min(int(alpha * n2), N))
        x.append(k)
        y_rd.append(None if proba_A == 0 else math.log10(proba_A))
        y_alpha.append(None if proba_B == 0 else math.log10(proba_B))
        # y_rd.append(proba_A)
        # y_alpha.append(proba_B)

    if args.csv:
        df = pd.DataFrame({"nb_mal": x, "proba_MtC": y_rd, "proba_MtC_alpha": y_alpha})
        df = df.round(3)
        # df = df.reindex(index=df.index[::-1])
        df.to_csv(args.csv, index=False, sep="\t", na_rep="nan")

    else:
        plt.figure()
        plt.plot(x, y_rd, label="Selection aléatoire")
        plt.plot(x, y_alpha, label=f"Sabine avec sous tirage aléatoire, Alpha = {alpha:.1f}, + gain={n2 - n}",
                 linestyle="--")
        plt.xlabel("Nombre de Malveillant")
        plt.ylabel("log10(Proba(alpha))")
        plt.title(f"Proba(alpha) en fonction du nombre de malveillant, n={n}, n2={n2}, P={min(int(alpha * n2), N)}")
        plt.xlim(4, N)
        plt.legend()
        plt.show()
