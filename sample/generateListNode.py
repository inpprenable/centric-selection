#!/usr/bin/env python3

import json
import argparse
from enum import Enum

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
        matrix = matrix.tolist()
        args.output.write(json.dumps(matrix))
        args.output.close()

