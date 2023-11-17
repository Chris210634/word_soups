#!/bin/bash
# Run generate_waffle_word_list.ipynb first, which saves waffle descriptors in cache/waffle_descriptors_512_count.list

python main_crossdataset.py --seed 1 --eval_only 1 --soup_eval 1 --descriptor_file cache/waffle_descriptors_512_count.list --shuffle_descriptors 1 --n_descriptors $1 --use_pretrained_image_features 1 --use_cached_image_features 1 

python main_crossdataset.py --seed 2 --eval_only 1 --soup_eval 1 --descriptor_file cache/waffle_descriptors_512_count.list --shuffle_descriptors 1 --n_descriptors $1 --use_pretrained_image_features 1 --use_cached_image_features 1 

python main_crossdataset.py --seed 3 --eval_only 1 --soup_eval 1 --descriptor_file cache/waffle_descriptors_512_count.list --shuffle_descriptors 1 --n_descriptors $1 --use_pretrained_image_features 1 --use_cached_image_features 1 