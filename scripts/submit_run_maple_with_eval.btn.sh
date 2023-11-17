#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

sh run_maple_with_eval.btn.sh 2.5 > run_maple_with_eval.btn.sh_2.5.o 
