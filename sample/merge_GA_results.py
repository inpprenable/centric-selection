import argparse
import os

import pandas as pd
from matplotlib import pyplot as plt


def is_directory(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory")


def moyenne_interquartile(group):
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    subset = group[(group >= q1) & (group <= q3)]
    return subset.mean()


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a parcours with Genetic Algorithme')
    parser.add_argument("input", type=is_directory, help="The adjacent matrix file")
    parser.add_argument("--prefix", type=str, default="grid_", help="The prefix of the csv files to read")
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("-e", "--extend", action="store_true", default=False, help="Add other data (std)")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_parser()

    list_files = [f for f in os.listdir(args.input) if f.startswith(args.prefix) and f.endswith(".csv")]
    list_id = [int(f[len(args.prefix):-4]) for f in list_files]

    list_df = []
    for i in list_id:
        df = pd.read_csv(os.path.join(args.input, f"{args.prefix}{i}.csv"))
        df["id"] = i
        list_df.append(df)
    df = pd.concat(list_df)
    if not args.extend:
        result = df.groupby('nb_val').agg({'min': 'mean', 'max': 'mean'}).reset_index()
    else:
        result = df.groupby('nb_val').agg(
            min_mean=('min', 'mean'),
            min_std=('min', 'std'),
            max_mean=('max', 'mean'),
            max_std=('max', 'std')
        ).reset_index()
    if args.output is not None:
        result.to_csv(args.output, index=False, sep=",")
    else:


        df_grouped_min = df.groupby('nb_val')['min'].apply(moyenne_interquartile).reset_index()
        df_grouped_max = df.groupby('nb_val')['max'].apply(moyenne_interquartile).reset_index()

        plt.figure(figsize=(10, 6))
        plt.scatter(df['nb_val'], df['min'], alpha=0.6, s=10)
        plt.scatter(df['nb_val'], df['max'], alpha=0.6, s=10)

        plt.scatter(result['nb_val'], result['min_mean'] if args.extend else result['min'], color='red', label='Mean Min', s=50)
        plt.scatter(df_grouped_min['nb_val'], df_grouped_min['min'], color='green', label='Interquartile Mean', s=50)

        plt.scatter(result['nb_val'], result['max_mean'] if args.extend else result['max'], color='red',
                    label='Mean Max', s=50)
        plt.scatter(df_grouped_max['nb_val'], df_grouped_max['max'], color='green', label='Interquartile Mean', s=50)

        plt.xlabel('nb_val')
        plt.ylabel('min')
        plt.title('Nuage de points : min en fonction de nb_val')
        plt.grid(True)
        plt.show()
        print(result)
