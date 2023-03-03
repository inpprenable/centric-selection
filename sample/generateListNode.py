#!/usr/bin/env python3
import argparse
import io
import json
from enum import Enum
import numpy as np

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
    args = parser.parse_args()
    return args


class OutputType(Enum):
    map = "map"
    matAdj = "matAdj"

    def __str__(self):
        return self.value


def write_matrix_file(file: io.TextIOWrapper, matrix: np.ndarray, sep=","):
    assert len(matrix.shape) == 2
    na, nb = matrix.shape
    assert na == nb
    for i in range(na):
        string_builder = ""
        for j in range(na - 1):
            string_builder += str(matrix[i, j]) + sep
        string_builder += str(matrix[i, -1]) + "\n"
        file.write(string_builder)


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
        if args.output.name.endswith(".json"):
            matrix = matrix.tolist()
            args.output.write(json.dumps(matrix))
            args.output.close()
        elif args.output.name.endswith(".csv"):
            write_matrix_file(args.output, matrix)
            args.output.close()
