#!/bin/bash

python main_crossdataset.py --adaptive_margin 0.1 --seed 1 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --adaptive_margin 0.1 --seed 2 --lr $1 --modelname $2 --d $3
python main_crossdataset.py --adaptive_margin 0.1 --seed 3 --lr $1 --modelname $2 --d $3