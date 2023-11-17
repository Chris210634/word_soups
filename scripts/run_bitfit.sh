#!/bin/bash

python main_crossdataset.py --seed 1 --bitfit 1 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --seed 2 --bitfit 1 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --seed 3 --bitfit 1 --text_prompt_depth 0 --lr $1 --modelname $2 --d $3