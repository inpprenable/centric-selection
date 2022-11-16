This project is the repository of various simulation programmes used in the conference paper *A centre-based validator selection approach for a scalable BFT blockchain*.

# Usage

## Generate the map
To generate a map of nodes, use `generateListNode.py` with the argument `map` like below. The size of the map is defined by options `-h` and `-w` and the number of nodes is defined by the option `-n`.

```bash
python sample/generateListNode.py map my_output_map.txt
```
The file `list_node_example_conf_paper.txt` is the map used in the article.

Sometimes, you want to extract the adjacency matrix from this map. This can be done by giving the argument `matAdj` to the programme `generateListNode.py`. A random map is generated, or an old one can be used with the option `--map`

```bash
python sample/generateListNode.py matAdj --map my_input_map.txt my_output_matAdj.txt
```

## Random Selection
Use `random_scan.py` to select random subset of validators in a map previously defined. The number of validator is varying according options `--min` to `--max`. The option `--gen` is used to repeat the selection and give the average node distance and the associated standard deviation. Use `-o` to store data in a file. 
In case of future iterations with the same output file, data are overwritten only if the new command provides greater iteration number. 

Example:
```bash
python sample/random_scan.py my_input_map.txt --gen 10 -o random_results.txt
```

## Selection using distance minimising Genetic Algorithms
Use `GA_scan.py` to obtain a selection of validators minimising (or maximising distance with the option `--worst`) with Genetics Algorithms. This selection is closed to the best (and the worst) selection. 
Population and number of generations of genetic algorithms are controlled with options `--pop` and `--gen`

Example:
```bash
python sample/GA_scan.py my_input_map.txt -o ga_results.csv
```

## Centric selection for multiple iterations

Use `center_scan` to use the centric selection detailed in the article. `--ellipse` controls the number of iterations before measurement (to reach the steady state), `--gen` controls the number of iterations of measurements (given data are the average). 
`--step` is used to define an interval in the scan process.
`--output` returns the results in a file.

Example: 
```bash
python sample/center_scan.py my_input_map.txt -o centric_results.csv --gen 10
```

## Graphical exhibit of centric selection 

To emphasise the proposal selection, a graphical representation is available with `center_graphical.py`. This graphical representation can be based on a given map of nodes with the option `--map`. Otherwise, a random map is generated.

The number of validator is set with `--val` and can vary around with a fixed period given by `--evValPeriod`. The number evolves according to a Gaussian random draw around the value with a standard deviation given by the option `--evValStd`. If the period is set to 0, the number of validator doesn't evolve.

The position of nodes can also vary every fixed period of time defined by `--evMapPeriod` (no variation if the period is equal to 0). If the position varies, then it varies according to a Gaussian distribution around the initial position, with a standard deviation given by the option `--evMapStd`.

Few iterations can be realised before the graphic representation to reach a steady state using the option `--ellipse`. The parameter $\mu$ can be set using `--mu`. A gif version of the representation can be saved using the option `--print` to a file named `animation.gif`.

Example:
```bash
python sample/center_graphical.py --map my_input_map.txt --gen 100 --print
```