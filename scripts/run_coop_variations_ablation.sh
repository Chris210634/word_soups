#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

# kgcoop M=1 M=2
python main_crossdataset.py --seed 1 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 4e-5 --loss kgcoop --init_lam 8.0 --shallow_prompt_init "a" > run_kgcoop_M1.o
python main_crossdataset.py --seed 2 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 4e-5 --loss kgcoop --init_lam 8.0 --shallow_prompt_init "a" >> run_kgcoop_M1.o
python main_crossdataset.py --seed 3 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 4e-5 --loss kgcoop --init_lam 8.0 --shallow_prompt_init "a" >> run_kgcoop_M1.o

python main_crossdataset.py --seed 1 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 4e-5 --loss kgcoop --init_lam 8.0 --shallow_prompt_init "a photo" > run_kgcoop_M2.o
python main_crossdataset.py --seed 2 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 4e-5 --loss kgcoop --init_lam 8.0 --shallow_prompt_init "a photo" >> run_kgcoop_M2.o
python main_crossdataset.py --seed 3 --use_cached_image_features 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 4e-5 --loss kgcoop --init_lam 8.0 --shallow_prompt_init "a photo" >> run_kgcoop_M2.o

# prograd M=1 M=2
python figures/python_scripts/main_crossdataset.prograd.py --use_cached_image_features 1 --loss clip --init_lam 1.0 --seed 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 128e-5 --shallow_prompt_init "a" > run_prograd_M1.o
python figures/python_scripts/main_crossdataset.prograd.py --use_cached_image_features 1 --loss clip --init_lam 1.0 --seed 2 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 128e-5 --shallow_prompt_init "a" >> run_prograd_M1.o
python figures/python_scripts/main_crossdataset.prograd.py --use_cached_image_features 1 --loss clip --init_lam 1.0 --seed 3 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 128e-5 --shallow_prompt_init "a" >> run_prograd_M1.o

python figures/python_scripts/main_crossdataset.prograd.py --use_cached_image_features 1 --loss clip --init_lam 1.0 --seed 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 128e-5 --shallow_prompt_init "a photo" > run_prograd_M2.o
python figures/python_scripts/main_crossdataset.prograd.py --use_cached_image_features 1 --loss clip --init_lam 1.0 --seed 2 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 128e-5 --shallow_prompt_init "a photo" >> run_prograd_M2.o
python figures/python_scripts/main_crossdataset.prograd.py --use_cached_image_features 1 --loss clip --init_lam 1.0 --seed 3 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1 --lr 128e-5 --shallow_prompt_init "a photo" >> run_prograd_M2.o

# proda num_descriptors=2, 4, 8, 16, 64
for M in 2 4 8 16 64;
do
    python figures/python_scripts/main_crossdataset.proda.py --use_cached_image_features 1 --loss proda --seed 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1  --lr 32e-5 --bs 128 --num_descriptors $M > run_proda_M$M.o
    python figures/python_scripts/main_crossdataset.proda.py --use_cached_image_features 1 --loss proda --seed 2 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1  --lr 32e-5 --bs 128 --num_descriptors $M >> run_proda_M$M.o
    python figures/python_scripts/main_crossdataset.proda.py --use_cached_image_features 1 --loss proda --seed 3 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 0 --text_prompt_depth 1  --lr 32e-5 --bs 128 --num_descriptors $M >> run_proda_M$M.o
done
