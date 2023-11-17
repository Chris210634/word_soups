#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

python main_crossdataset.py --seed 1 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --train_with_descriptors 1 --soup_eval 1 --init_lam 0.25 --teacher_temp 10 --score_averaging 1 --lr 0.025 > run_maple_regularized_output_lam_0.25_temp_10.o

python main_crossdataset.py --seed 2 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --train_with_descriptors 1 --soup_eval 1 --init_lam 0.25 --teacher_temp 10 --score_averaging 1 --lr 0.025 >> run_maple_regularized_output_lam_0.25_temp_10.o

python main_crossdataset.py --seed 3 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --train_with_descriptors 1 --soup_eval 1 --init_lam 0.25 --teacher_temp 10 --score_averaging 1 --lr 0.025 >> run_maple_regularized_output_lam_0.25_temp_10.o