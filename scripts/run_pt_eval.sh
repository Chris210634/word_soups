#!/bin/bash

python main_crossdataset.py --seed 1 --modelname $2 --pretrained $3 --d $4 --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1 --score_averaging 1 --eval_only 1 > run_pt_eval_$2_$3.o
python main_crossdataset.py --seed 2 --modelname $2 --pretrained $3 --d $4 --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1 --score_averaging 1 --eval_only 1 >> run_pt_eval_$2_$3.o
python main_crossdataset.py --seed 3 --modelname $2 --pretrained $3 --d $4 --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1 --score_averaging 1 --eval_only 1 >> run_pt_eval_$2_$3.o