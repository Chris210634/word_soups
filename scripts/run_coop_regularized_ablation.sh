#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

for N in 2 4 8 16 32 64;
do

    python main_crossdataset.py --seed 1 --train_with_descriptors 1 --init_lam 0.25 --teacher_temp 10 --score_averaging 1 --soup_eval 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --use_cached_image_features 1 --descriptor_file cache/word_soup_64/word_soup_descriptors_seed1__ViT-B-16_openai.list --shuffle_descriptors 1 --n_descriptors $N > run_coop_regularized_M$N.o

    python main_crossdataset.py --seed 2 --train_with_descriptors 1 --init_lam 0.25 --teacher_temp 10 --score_averaging 1 --soup_eval 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --use_cached_image_features 1 --descriptor_file cache/word_soup_64/word_soup_descriptors_seed2__ViT-B-16_openai.list --shuffle_descriptors 1 --n_descriptors $N >> run_coop_regularized_M$N.o

    python main_crossdataset.py --seed 3 --train_with_descriptors 1 --init_lam 0.25 --teacher_temp 10 --score_averaging 1 --soup_eval 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 8e-5 --use_cached_image_features 1 --descriptor_file cache/word_soup_64/word_soup_descriptors_seed3__ViT-B-16_openai.list --shuffle_descriptors 1 --n_descriptors $N >> run_coop_regularized_M$N.o

done