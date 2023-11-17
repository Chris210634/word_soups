#!/bin/bash

python main_crossdataset.py --seed 1 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --seed 2 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --seed 3 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3