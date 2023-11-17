#!/bin/bash

python main_crossdataset.py --seed 1 --train_with_descriptors 1 --soup_eval 1 --init_lam $1 --teacher_temp $2 --score_averaging 1 --soup_eval 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --loss clip > run_coop_regularized_output_lam_$1_temp_$2_clip_loss.o

python main_crossdataset.py --seed 2 --train_with_descriptors 1 --soup_eval 1 --init_lam $1 --teacher_temp $2 --score_averaging 1 --soup_eval 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --loss clip >> run_coop_regularized_output_lam_$1_temp_$2_clip_loss.o

python main_crossdataset.py --seed 3 --train_with_descriptors 1 --soup_eval 1 --init_lam $1 --teacher_temp $2 --score_averaging 1 --soup_eval 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --loss clip >> run_coop_regularized_output_lam_$1_temp_$2_clip_loss.o