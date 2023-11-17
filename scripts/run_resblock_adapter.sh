#!/bin/bash

python main_crossdataset.py --seed 1 --resblock_adapter 1 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --seed 2 --resblock_adapter 1 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --seed 3 --resblock_adapter 1 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3