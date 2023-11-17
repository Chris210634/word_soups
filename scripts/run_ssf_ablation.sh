#!/bin/bash

for T in 10 11;
do
    python main_crossdataset.py --seed 1 --ssf 1 --text_prompt_depth 0 --lr 0.0001 --layer_start_v $T --layer_start_t $T > run_ssf_M$T.o
    python main_crossdataset.py --seed 2 --ssf 1 --text_prompt_depth 0 --lr 0.0001 --layer_start_v $T --layer_start_t $T >> run_ssf_M$T.o
    python main_crossdataset.py --seed 3 --ssf 1 --text_prompt_depth 0 --lr 0.0001 --layer_start_v $T --layer_start_t $T >> run_ssf_M$T.o
done