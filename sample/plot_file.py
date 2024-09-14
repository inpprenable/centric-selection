import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def is_directory(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory")


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solve a parcours with Genetic Algorithme')
    parser.add_argument("-i", "--input", type=is_directory, help="The adjacent matrix file")
    parser.add_argument("--prefix", type=str, default="results_", help="The prefix of the csv files to read")
    parser.add_argument("-o", "--output", type=str,
                        help="the file to store the results", default=None)
    parser.add_argument("-e", "--extend", action="store_true", default=False, help="Add other data (std)")
    parser.add_argument("--csv", action="store_true", default=False, help="Export data in a csv format")
    parser.add_argument("--files", type=str, nargs='+', help="The files to read")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_parser()

    # Initialisation de la figure pour le plot

    if args.files is not None:
        file_list = args.files
    else:
        file_list = glob.glob(f'{args.input}/{args.prefix}*.csv')

        file_list = sorted(file_list)
        file_label = [
            (file, float(file.split('/')[-1].split('.csv')[0].split(args.prefix)[1])) for file in file_list
        ]
        sorted_file_label = sorted(file_label, key=lambda x: x[1])
        file_list = [file[0] for file in sorted_file_label]

    if len(file_list) == 0:
        print(f"No file found with prefix {args.prefix}")
        exit(1)

    if args.csv:

        # Initialisation d'un DataFrame vide pour commencer la fusion
        merged_df = pd.DataFrame()
        # Parcourir chaque fichier dans la liste
        for file in file_list:
            # Lire le fichier CSV
            df = pd.read_csv(file)

            # Extraire le nom du fichier sans le chemin ni l'extension
            if args.files is not None:
                label = file.split('/')[-1].split('.csv')[0]
            else:
                label = file.split('/')[-1].split('.csv')[0].split(args.prefix)[1]

            if args.extend:
                # Renommer la colonne avg_weight avec le nom du fichier
                df = df[['nb_val', 'avg_weight', 'std']].rename(
                    columns={'avg_weight': f'{label}', 'std': f'{label}_std'})

                # Fusionner avec le DataFrame final
                if merged_df.empty:
                    merged_df = df  # Si le DataFrame est vide, initialisez-le
                else:
                    merged_df = pd.merge(merged_df, df, on='nb_val', how='outer')  # Fusion sur nb_val
            else:
                # Renommer la colonne avg_weight avec le nom du fichier
                df = df[['nb_val', 'avg_weight']].rename(columns={'avg_weight': f'{label}'})

                # Fusionner avec le DataFrame final
                if merged_df.empty:
                    merged_df = df  # Si le DataFrame est vide, initialisez-le
                else:
                    merged_df = pd.merge(merged_df, df, on='nb_val', how='outer')  # Fusion sur nb_val

        if args.output is not None:
            new_file_name = args.output.split(".csv")[0] + "_weight.csv"
            merged_df.to_csv(new_file_name, index=False, sep="\t")
        else:
            print(merged_df)

    else:
        plt.figure(figsize=(10, 6))
        # Parcourir chaque fichier dans la liste
        for file in file_list:
            # Lire le fichier CSV
            df = pd.read_csv(file)

            # Extraire le nom du fichier sans le chemin ni l'extension
            label = file.split('/')[-1].split('.csv')[0]

            if args.extend:
                # Tracer la courbe avec des barres d'erreur
                plt.errorbar(df['nb_val'], df['avg_weight'], yerr=df['std'], fmt='-o', label=label, capsize=5)
            else:
                # Tracer la courbe sans les barres d'erreur
                plt.plot(df['nb_val'], df['avg_weight'], '-o', label=label)

        # Ajouter des légendes et titres
        plt.xlabel('nb_val')
        plt.ylabel('avg_weight')
        plt.title('avg_weight vs nb_val avec erreurs standard')
        plt.legend()
        plt.grid(True)

        if args.output is not None:
            list_elements = args.output.split(".")
            plt.savefig(list_elements[0] + "_weight." + list_elements[1])
        else:
            # Afficher le graphique
            plt.show()

    if args.csv:
        # Initialisation d'un DataFrame vide pour commencer la fusion
        merged_df = pd.DataFrame()
        # Parcourir chaque fichier dans la liste
        for file in file_list:
            # Lire le fichier CSV
            df = pd.read_csv(file)

            # Extraire le nom du fichier sans le chemin ni l'extension
            if args.files is not None:
                label = file.split('/')[-1].split('.csv')[0]
            else:
                label = file.split('/')[-1].split('.csv')[0].split(args.prefix)[1]

            # Renommer la colonne avg_weight avec le nom du fichier
            df = df[['nb_val', 'gini_coef_r']].rename(
                columns={'gini_coef_r': f'{label}'})

            # Fusionner avec le DataFrame final
            if merged_df.empty:
                merged_df = df  # Si le DataFrame est vide, initialisez-le
            else:
                merged_df = pd.merge(merged_df, df, on='nb_val', how='outer')  # Fusion sur nb_val

        if args.output is not None:
            new_file_name = args.output.split(".csv")[0] + "_gini.csv"
            merged_df.to_csv(new_file_name, index=False, sep="\t")
        else:
            print(merged_df)
    else:

        # Initialisation de la figure pour le plot
        plt.figure(figsize=(10, 6))

        # Parcourir chaque fichier dans la liste
        for file in file_list:
            # Lire le fichier CSV
            df = pd.read_csv(file)

            # Extraire le nom du fichier sans le chemin ni l'extension
            label = file.split('/')[-1].split('.csv')[0]

            plt.plot(df['nb_val'], df['gini_coef_r'], '-o', label=label)

        # Ajouter des légendes et titres
        plt.xlabel('nb_val')
        plt.ylabel('gini_coef_r')
        plt.title('gini_coef_r vs nb_val')
        plt.legend()
        plt.grid(True)

        if args.output is not None:
            list_elements = args.output.split(".")
            plt.savefig(list_elements[0] + "_gini." + list_elements[1])
        else:
            # Afficher le graphique
            plt.show()
