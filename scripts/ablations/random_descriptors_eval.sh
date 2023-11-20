#!/bin/bash

python main_crossdataset.py --seed 1 --eval_only 1 --soup_eval 1 --token_offset_eval 1 --descriptor_file descriptions.list --shuffle_descriptors 1 --n_descriptors $1 --use_pretrained_image_features 1 --use_cached_image_features 1 

python main_crossdataset.py --seed 2 --eval_only 1 --soup_eval 1 --token_offset_eval 1 --descriptor_file descriptions.list --shuffle_descriptors 1 --n_descriptors $1 --use_pretrained_image_features 1 --use_cached_image_features 1 

python main_crossdataset.py --seed 3 --eval_only 1 --soup_eval 1 --token_offset_eval 1 --descriptor_file descriptions.list --shuffle_descriptors 1 --n_descriptors $1 --use_pretrained_image_features 1 --use_cached_image_features 1 