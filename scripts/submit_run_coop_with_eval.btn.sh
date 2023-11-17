#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

sh run_coop_with_eval.btn.sh 0.0002 > run_coop_with_eval.btn.sh_0.0002.o 
