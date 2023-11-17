#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

for M in 8 16 32 64 128;
do
    python main_crossdataset.py --seed 1 --adapter 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 0 --lr 0.006 --rank $M > run_adapter_M$M.o
    python main_crossdataset.py --seed 2 --adapter 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 0  --lr 0.006 --rank $M >> run_adapter_M$M.o
    python main_crossdataset.py --seed 3 --adapter 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 0  --lr 0.006 --rank $M >> run_adapter_M$M.o
done