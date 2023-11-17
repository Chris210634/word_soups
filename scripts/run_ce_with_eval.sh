#!/bin/bash

python main_crossdataset.py --seed 1 --lr $1 --modelname $2 --d $3 --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1
python main_crossdataset.py --seed 2 --lr $1 --modelname $2 --d $3 --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1
python main_crossdataset.py --seed 3 --lr $1 --modelname $2 --d $3 --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1