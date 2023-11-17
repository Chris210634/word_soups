#!/bin/bash

python main_crossdataset.py --seed 1 --use_cached_image_features 1 --eval_only 1 --score_averaging 1 --descriptor_file cache/word_soup_64/word_soup_descriptors_seed1__ViT-B-16_openai.list --shuffle_descriptors 1 --n_descriptors $1
python main_crossdataset.py --seed 2 --use_cached_image_features 1 --eval_only 1 --score_averaging 1 --descriptor_file cache/word_soup_64/word_soup_descriptors_seed2__ViT-B-16_openai.list --shuffle_descriptors 1 --n_descriptors $1
python main_crossdataset.py --seed 3 --use_cached_image_features 1 --eval_only 1 --score_averaging 1 --descriptor_file cache/word_soup_64/word_soup_descriptors_seed3__ViT-B-16_openai.list --shuffle_descriptors 1 --n_descriptors $1