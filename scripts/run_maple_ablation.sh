#!/bin/bash -l
mkdir /scratch/cliao25
module load python3/3.8.10 pytorch/1.13.1
sh copy_data.sh

python main_crossdataset.py --seed 1 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --lr 0.025 --shallow_prompt_init "a" --visual_prompt_length 1 --text_prompt_length 1 > run_maple_M1.o
python main_crossdataset.py --seed 2 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --lr 0.025 --shallow_prompt_init "a" --visual_prompt_length 1 --text_prompt_length 1 >> run_maple_M1.o
python main_crossdataset.py --seed 3 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --lr 0.025 --shallow_prompt_init "a" --visual_prompt_length 1 --text_prompt_length 1 >> run_maple_M1.o

python main_crossdataset.py --seed 1 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --lr 0.025 --shallow_prompt_init "a photo" --visual_prompt_length 2 --text_prompt_length 2 > run_maple_M2.o
python main_crossdataset.py --seed 2 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --lr 0.025 --shallow_prompt_init "a photo" --visual_prompt_length 2 --text_prompt_length 2 >> run_maple_M2.o
python main_crossdataset.py --seed 3 --prompt_lr_multi 1 --maple 1 --train_text_encoder 0 --train_visual_encoder 0 --visual_prompt_depth 3 --text_prompt_depth 3 --lr 0.025 --shallow_prompt_init "a photo" --visual_prompt_length 2 --text_prompt_length 2 >> run_maple_M2.o