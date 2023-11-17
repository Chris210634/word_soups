#!/bin/bash

for seed in 1 2 3;

do
    python main_novelclasses.py --adaptive_margin 0.1 --dataset ImageNet \
    --n_epochs 1 --iters_per_epoch 1000 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset Caltech101 \
    --n_epochs 1 --iters_per_epoch 500 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset OxfordPets \
    --n_epochs 1 --iters_per_epoch 500 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset StanfordCars \
    --n_epochs 1 --iters_per_epoch 500 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset Flowers102 \
    --n_epochs 1 --iters_per_epoch 500 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset Food101 \
    --n_epochs 1 --iters_per_epoch 250 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset FGVCAircraft \
    --n_epochs 1 --iters_per_epoch 500 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset SUN397 \
    --n_epochs 1 --iters_per_epoch 500 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset DTD \
    --n_epochs 1 --iters_per_epoch 500 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset EuroSAT \
    --n_epochs 1 --iters_per_epoch 250 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1

    python main_novelclasses.py --adaptive_margin 0.1 --dataset UCF101 \
    --n_epochs 1 --iters_per_epoch 1000 --lr $1 --seed $seed \
    --openai_eval 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --soup_eval 1 --token_offset_eval 1
    
done