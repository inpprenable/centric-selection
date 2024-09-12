import argparse
import os

import pandas as pd


def is_directory(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory")


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
        result.to_csv(args.output, index=False, sep="\t")
    else:
        print(result)
