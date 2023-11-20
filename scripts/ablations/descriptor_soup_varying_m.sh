#!/bin/bash

python main_crossdataset.py --seed 1 --use_cached_image_features 1 --eval_only 1 --soup_eval 1 --token_offset_eval 1 --descriptor_file cache/good_descriptions_seed1__ViT-B-16_openai.list --n_descriptors $1

python main_crossdataset.py --seed 2 --use_cached_image_features 1 --eval_only 1 --soup_eval 1 --token_offset_eval 1 --descriptor_file cache/good_descriptions_seed1__ViT-B-16_openai.list --n_descriptors $1

python main_crossdataset.py --seed 3 --use_cached_image_features 1 --eval_only 1 --soup_eval 1 --token_offset_eval 1 --descriptor_file cache/good_descriptions_seed1__ViT-B-16_openai.list --n_descriptors $1