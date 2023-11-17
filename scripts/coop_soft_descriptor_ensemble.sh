#!/bin/bash 

python train_soft_descriptors.py --lr 8e-5 \
--descriptor_file figures/8_random_10_token_word_chains_seed1.list \
--seed 1 --iters_per_epoch 750  --n_epochs 1 > train_softd.o

python train_soft_descriptors.py --lr 8e-5 \
--descriptor_file figures/8_random_10_token_word_chains_seed2.list \
--seed 2 --iters_per_epoch 750  --n_epochs 1 >> train_softd.o

python train_soft_descriptors.py --lr 8e-5 \
--descriptor_file figures/8_random_10_token_word_chains_seed3.list \
--seed 3 --iters_per_epoch 750  --n_epochs 1 >> train_softd.o