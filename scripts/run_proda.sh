#!/bin/bash

python figures/python_scripts/main_crossdataset.proda.py --use_cached_image_features 1 --loss proda --seed 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3 --bs 128

python figures/python_scripts/main_crossdataset.proda.py --use_cached_image_features 1 --loss proda --seed 2 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3 --bs 128

python figures/python_scripts/main_crossdataset.proda.py --use_cached_image_features 1 --loss proda --seed 3 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3 --bs 128