#!/bin/bash
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

sh descriptor_soup_varying_m.sh 1 > descriptor_soup_1.o
sh descriptor_soup_varying_m.sh 2 > descriptor_soup_2.o
sh descriptor_soup_varying_m.sh 4 > descriptor_soup_4.o
sh descriptor_soup_varying_m.sh 8 > descriptor_soup_8.o

python get_greedy_word_soup.py --dataset ImageNet --seed 1 --n_descriptors 64
python get_greedy_word_soup.py --dataset ImageNet --seed 2 --n_descriptors 64
python get_greedy_word_soup.py --dataset ImageNet --seed 3 --n_descriptors 64

sh word_soup_varying_m.sh 1 > word_soup_1.o
sh word_soup_varying_m.sh 2 > word_soup_2.o
sh word_soup_varying_m.sh 4 > word_soup_4.o
sh word_soup_varying_m.sh 8 > word_soup_8.o
sh word_soup_varying_m.sh 16 > word_soup_16.o
sh word_soup_varying_m.sh 32 > word_soup_32.o
sh word_soup_varying_m.sh 64 > word_soup_64.o