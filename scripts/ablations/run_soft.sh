#!/bin/bash
python main_crossdataset.soft.py --eval_only 1  --seed 1 --descriptor_file cache/soft_descriptors/12_token_random_word_chain/word_soup_descriptors_seed{}__ViT-B-16_openai.list.soft > eval_soft_12_token_random_word_chain.o
for epoch in 0 1 2 3 4 5 6 7 8 9
do 
    python main_crossdataset.soft.py --eval_only 1  --seed 1 --descriptor_file cache/soft_descriptors/12_token_random_word_chain/word_soup_descriptors_seed{}__ViT-B-16_openai.list_e$epoch.soft >> eval_soft_12_token_random_word_chain.o
done

# for evaluating SoOp soft descriptor ensemble:
# ! python main_crossdataset.soft.py --eval_only 1  --seed 1 \
# --descriptor_file cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed{}.list_e0.soft \
# > coop_soft_descriptor_ensemble_of_8_eval.o
# ! python main_crossdataset.soft.py --eval_only 1  --seed 2 \
# --descriptor_file cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed{}.list_e0.soft \
# >> coop_soft_descriptor_ensemble_of_8_eval.o
# ! python main_crossdataset.soft.py --eval_only 1  --seed 3 \
# --descriptor_file cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed{}.list_e0.soft \
# >> coop_soft_descriptor_ensemble_of_8_eval.o