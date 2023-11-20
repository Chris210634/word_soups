#!/bin/bash

python main_crossdataset.py --seed 1 --train_text_encoder 0 --train_visual_encoder 0 \
--visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 \
--score_averaging 1 --soup_eval 1 --label_smoothing 0.25 > coop_label_smoothing.o

python main_crossdataset.py --seed 2 --train_text_encoder 0 --train_visual_encoder 0 \
--visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 \
--score_averaging 1 --soup_eval 1 --label_smoothing 0.25 >> coop_label_smoothing.o

python main_crossdataset.py --seed 3 --train_text_encoder 0 --train_visual_encoder 0 \
--visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 \
--score_averaging 1 --soup_eval 1 --label_smoothing 0.25 >> coop_label_smoothing.o