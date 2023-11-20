#!/bin/bash

python main_crossdataset.py --seed 1 --use_cached_image_features 1 --eval_only 1 --openai_eval 1 --descriptor_file descriptions.list --n_descriptors $1