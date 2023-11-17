#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

for M in 8 4 2 1;
do
    python main_crossdataset.py --seed 1 --lora 1 --text_prompt_depth 0 --lr 0.00001 --rank $M > run_lora_M$M.o
    python main_crossdataset.py --seed 2 --lora 1 --text_prompt_depth 0 --lr 0.00001 --rank $M >> run_lora_M$M.o
    python main_crossdataset.py --seed 3 --lora 1 --text_prompt_depth 0 --lr 0.00001 --rank $M >> run_lora_M$M.o
done