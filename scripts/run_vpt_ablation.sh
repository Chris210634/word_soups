#!/bin/bash

for M in 3 2 1;
do
    python main_crossdataset.py --seed 1 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 0 --lr 0.8 --visual_prompt_length $M > run_vpt_deep_M$M.o
    python main_crossdataset.py --seed 2 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 0 --lr 0.8 --visual_prompt_length $M >> run_vpt_deep_M$M.o
    python main_crossdataset.py --seed 3 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 0 --lr 0.8 --visual_prompt_length $M >> run_vpt_deep_M$M.o
done

for M in 3 2 1;
do
    python main_crossdataset.py --seed 1 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 1 --text_prompt_depth 0 --lr 0.8 --visual_prompt_length $M > run_vpt_shallow_M$M.o
    python main_crossdataset.py --seed 2 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 1 --text_prompt_depth 0 --lr 0.8 --visual_prompt_length $M >> run_vpt_shallow_M$M.o
    python main_crossdataset.py --seed 3 --prompt_lr_multi 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 1 --text_prompt_depth 0 --lr 0.8 --visual_prompt_length $M >> run_vpt_shallow_M$M.o
done