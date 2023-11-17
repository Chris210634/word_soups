#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

for T in 10 11;
do
    python main_crossdataset.py --seed 1 --ssf 1 --text_prompt_depth 0 --lr 0.0001 --layer_start_v $T --layer_start_t $T > run_ssf_M$T.o
    python main_crossdataset.py --seed 2 --ssf 1 --text_prompt_depth 0 --lr 0.0001 --layer_start_v $T --layer_start_t $T >> run_ssf_M$T.o
    python main_crossdataset.py --seed 3 --ssf 1 --text_prompt_depth 0 --lr 0.0001 --layer_start_v $T --layer_start_t $T >> run_ssf_M$T.o
done