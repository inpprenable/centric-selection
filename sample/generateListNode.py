#!/usr/bin/env python3
import argparse
import io
import json
from enum import Enum

import numpy as np
import pandas as pd

from fonctionGraph import generate_node, calcul_matrice_adjacente


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a random map')
    parser.add_argument("type", type=OutputType, choices=list(OutputType),
                        help="Generate a position list of Node map or the adjacency matrix of distances between "
                             "this nodes")
    parser.add_argument("output", type=argparse.FileType('w+', encoding='utf-8'), help="the file to store the map")
    parser.add_argument("--map", type=argparse.FileType('r', encoding='utf-8'), help="the file to store the map",
                        default=None)
    parser.add_argument("-H", "--height", type=int, default=50,
                        help="Set the height")
    parser.add_argument("-w", "--width", type=int, default=50,
                        help="Set the width")
    parser.add_argument("-n", "--nbnode", type=int, default=200,
                        help="Set the number of nodes")
    parser.add_argument("--norm", action="store_true", default=False,
                        help="Normalize the adjacency matrix according to the GCN method")
    args = parser.parse_args()
    return args


class OutputType(Enum):
    map = "map"
    matAdj = "matAdj"

    def __str__(self):
        return self.value


def write_matrix_file(file: io.TextIOWrapper, matrix: np.ndarray, sep=","):
    df = pd.DataFrame(matrix)
    df.to_csv(file, index=False, header=False, sep=sep)


def normalize_max(w:np.ndarray) -> np.ndarray:
    # ignorer la diagonale forcée à 0
    w_no_diag = w.copy()
    np.fill_diagonal(w_no_diag, np.nan)
    w_max = np.nanmax(w_no_diag)
    w_norm = w / w_max
    np.fill_diagonal(w_norm, 0.0)
    return w_norm


if __name__ == '__main__':
    args = create_parser()

    if args.type == OutputType.map:
        list_node = generate_node(args.height, args.width, args.nbnode)
        args.output.write(json.dumps(list_node))
        args.output.close()
    else:
        list_node = None
        if args.map is None:
            list_node = generate_node(args.height, args.width, args.nbnode)
        else:
            list_node = json.loads(args.map.read())
            args.map.close()
        list_node = {int(k): v for k, v in list_node.items()}
        matrix = calcul_matrice_adjacente(list_node)
        if args.norm:
            matrix = normalize_max(matrix)
        if args.output.name.endswith(".json"):
            matrix = matrix.tolist()
            args.output.write(json.dumps(matrix))
            args.output.close()
        elif args.output.name.endswith(".csv"):
            write_matrix_file(args.output, matrix)
            args.output.close()
