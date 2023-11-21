#!/bin/bash

python figures/python_scripts/main_crossdataset.prograd.py --loss clip --init_lam 1.0 --seed 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3

python figures/python_scripts/main_crossdataset.prograd.py --loss clip --init_lam 1.0 --seed 2 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3

python figures/python_scripts/main_crossdataset.prograd.py --loss clip --init_lam 1.0 --seed 3 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3