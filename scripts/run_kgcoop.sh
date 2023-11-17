#!/bin/bash

python main_crossdataset.py --seed 1 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3 --loss kgcoop --init_lam 8.0
python main_crossdataset.py --seed 2 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3 --loss kgcoop --init_lam 8.0
python main_crossdataset.py --seed 3 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr $1 --modelname $2 --d $3 --loss kgcoop --init_lam 8.0