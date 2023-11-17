#!/bin/bash -l

python get_greedy_descriptor_soup.py --dataset $1 --seed 1 --subsample_classes base
python get_greedy_descriptor_soup.py --dataset $1 --seed 2 --subsample_classes base
python get_greedy_descriptor_soup.py --dataset $1 --seed 3 --subsample_classes base