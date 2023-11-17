#!/bin/bash

# This script handles anything that is not handled by scripts/run_pt_eval.sh
# you should first calculate the word soup for m=32 and run scripts/run_pt_eval.sh
# That will give you the data for ZS, openai ensemble, GPT, descriptor soup, and 32-word-soup

# This file handles the m=8 word soup and waffle soup

# see also waffle_descriptors_eval.sh

python main_crossdataset.py --seed 1 --eval_only 1 --soup_eval 1 --descriptor_file cache/waffle_descriptors_512_count.list --shuffle_descriptors 1 --n_descriptors 16 --modelname $2 --pretrained $3 --d $4 > run_waffle_eval_$2_$3.o
python main_crossdataset.py --seed 2 --eval_only 1 --soup_eval 1 --descriptor_file cache/waffle_descriptors_512_count.list --shuffle_descriptors 1 --n_descriptors 16 --modelname $2 --pretrained $3 --d $4 >> run_waffle_eval_$2_$3.o
python main_crossdataset.py --seed 3 --eval_only 1 --soup_eval 1 --descriptor_file cache/waffle_descriptors_512_count.list --shuffle_descriptors 1 --n_descriptors 16 --modelname $2 --pretrained $3 --d $4 >> run_waffle_eval_$2_$3.o

python main_crossdataset.py --seed 1 --eval_only 1 --score_averaging 1 --descriptor_file cache/word_soup_descriptors_seed1__$2_$3.list --shuffle_descriptors 1 --n_descriptors 8 --modelname $2 --pretrained $3 --d $4 > run_wordsoup8_eval_$2_$3.o
python main_crossdataset.py --seed 2 --eval_only 1 --score_averaging 1 --descriptor_file cache/word_soup_descriptors_seed2__$2_$3.list --shuffle_descriptors 1 --n_descriptors 8 --modelname $2 --pretrained $3 --d $4 >> run_wordsoup8_eval_$2_$3.o
python main_crossdataset.py --seed 3 --eval_only 1 --score_averaging 1 --descriptor_file cache/word_soup_descriptors_seed3__$2_$3.list --shuffle_descriptors 1 --n_descriptors 8 --modelname $2 --pretrained $3 --d $4 >> run_wordsoup8_eval_$2_$3.o