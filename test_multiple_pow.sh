#!/bin/bash


#for pow in 0.5 0.6 0.7 0.8 0.9 1 1.5 2 ; do
#  mkdir -p "pow_study/pow_$pow"
#  for stake in neutral random low_stake high_stake ; do
#    python sample/center_stake_bal.py example/list_node_example_conf_paper.json --gen 2000 --horizon 0 -s stake_exp/$stake.json --pow $pow -o "pow_study/pow_$pow/test_$stake.csv"
#    done
#done
#    python sample/center_stake_bal.py example/list_node_example_conf_paper.json --gen 2000 --horizon 200 -s stake_exp/$stake.json -o "pow_study_fun_test/test_$stake.csv"

for stake in neutral random low_stake high_stake ; do
  python sample/center_stake_bal.py example/list_node_example_conf_paper.json -s stake_exp/$stake.json --horizon 0 --gen 2000 --elipse 200 -o new_stake_exp/$stake.csv
done

#for val in 10 50 100 150 ; do
#  python sample/center_stake.py example/list_node_example_conf_paper.json --gen 2000 --horizon 0 -s stake_exp/low_stake.json --nb $val -o "pow_study_fun_2/low_val_$val.csv"
#done