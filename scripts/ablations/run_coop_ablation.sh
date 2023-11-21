#!/bin/bash

python main_crossdataset.py --seed 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 > coop_M3.o
python main_crossdataset.py --seed 2 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 >> coop_M3.o
python main_crossdataset.py --seed 3 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 >> coop_M3.o

python main_crossdataset.py --seed 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --shallow_prompt_init "a" > coop_M1.o
python main_crossdataset.py --seed 2 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --shallow_prompt_init "a" >> coop_M1.o
python main_crossdataset.py --seed 3 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --shallow_prompt_init "a" >> coop_M1.o

python main_crossdataset.py --seed 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --shallow_prompt_init "a photo" > coop_M2.o
python main_crossdataset.py --seed 2 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --shallow_prompt_init "a photo" >> coop_M2.o
python main_crossdataset.py --seed 3 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --shallow_prompt_init "a photo" >> coop_M2.o