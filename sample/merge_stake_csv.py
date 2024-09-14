import argparse

import numpy as np
import pandas as pd

from sample.stake_gen import read_stake_json


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a parcours with Genetic Algorithme')
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("files", type=argparse.FileType("r"), nargs='+', help="The files to read")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_parser()
    df = None
    for file in args.files:
        label = file.name.split('/')[-1].split('.json')[0]
        stake = read_stake_json(file)
        index = np.arange(len(stake)) + 1
        df_temp = pd.DataFrame({label: stake, "nb_val": index})
        if df is None:
            df = df_temp
        else:
            df = pd.merge(df, df_temp, on="nb_val", how="outer")
        # df = pd.concat([df, df_temp], axis=1)
    df.set_index("nb_val", inplace=True)

    if args.output is not None:
        df.to_csv(args.output, index=True, sep="\t")
    else:
        print(df.head())
