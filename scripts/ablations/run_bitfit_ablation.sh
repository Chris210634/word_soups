#!/bin/bash

for T in 10 11;
do
    python main_crossdataset.py --seed 1 --bitfit 1 --text_prompt_depth 0 --lr 0.000125 --layer_start_v $T --layer_start_t $T > run_bitfit_M$T.o
    python main_crossdataset.py --seed 2 --bitfit 1 --text_prompt_depth 0 --lr 0.000125 --layer_start_v $T --layer_start_t $T >> run_bitfit_M$T.o
    python main_crossdataset.py --seed 3 --bitfit 1 --text_prompt_depth 0 --lr 0.000125 --layer_start_v $T --layer_start_t $T >> run_bitfit_M$T.o
done