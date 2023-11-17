#!/bin/bash

python main_crossdataset.py --seed 1 --train_with_descriptors 1 --soup_eval 1 --init_lam $1 --teacher_temp $2 --score_averaging 1 > run_ce_regularized_output_lam_$1_temp_$2.o

python main_crossdataset.py --seed 2 --train_with_descriptors 1 --soup_eval 1 --init_lam $1 --teacher_temp $2 --score_averaging 1 >> run_ce_regularized_output_lam_$1_temp_$2.o

python main_crossdataset.py --seed 3 --train_with_descriptors 1 --soup_eval 1 --init_lam $1 --teacher_temp $2 --score_averaging 1 >> run_ce_regularized_output_lam_$1_temp_$2.o