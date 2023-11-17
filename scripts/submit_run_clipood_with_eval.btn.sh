#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

sh run_clipood_with_eval.btn.sh 5e-05 > run_clipood_with_eval.btn.sh_5e-05.o 
