#!/bin/bash

python main_crossdataset.py --seed 1 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --seed 2 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --seed 3 --lr $1 --modelname $2 --d $3