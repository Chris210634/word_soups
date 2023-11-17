#!/bin/bash
for epoch in 0 1 2 3 4 5 6 7 8 9
do 
    python main_crossdataset.soft.py --eval_only 1  --seed 1 --descriptor_file cache/soft_descriptors/word_soup_descriptors_seed{}__ViT-B-16_openai.list_e$epoch.soft > eval_e$epoch.o
    python main_crossdataset.soft.py --eval_only 1  --seed 2 --descriptor_file cache/soft_descriptors/word_soup_descriptors_seed{}__ViT-B-16_openai.list_e$epoch.soft >> eval_e$epoch.o
    python main_crossdataset.soft.py --eval_only 1  --seed 3 --descriptor_file cache/soft_descriptors/word_soup_descriptors_seed{}__ViT-B-16_openai.list_e$epoch.soft >> eval_e$epoch.o
done