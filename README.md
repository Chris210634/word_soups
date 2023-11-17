# Word and Descriptor Soups 🍜
-----------------------------------------------------

Code in this repo uses code from [multimodal prompt learning](https://github.com/muzairkhattak/multimodal-prompt-learning), which in turn uses code from [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp).

## ⏳ Installation
-------------------

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Create a directory somewhere called `data/`. Download all 15 zip files from the shared Google Drive and unzip them into `data/`. The resulting file tree should look like:
```
data/
|-- caltech-101
|-- dtd
|-- eurosat
|-- fgvc_aircraft
|-- food-101
|-- imagenet
|-- imagenet-adversarial
|-- imagenet-rendition
|-- imagenet-sketch
|-- imagenetv2
|-- oxford_flowers
|-- oxford_pets
|-- stanford_cars
|-- sun397
|-- ucf101
```

Alternatively, follow the download instructions here (some dataset links are stale):
[installing datasets](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/DATASETS.md)

## 🍜 Descriptor soups
---------------------------

### (1) Generate Description Features
First, calculate the descriptor features on ImageNet.
Use `preprocess/generate_description_features.py`.
This python file reads from `preprocess/descriptions.list`, 
which is a sorted list of 4227 unique GPT descriptors. They begin with a space and end in a period.
Currently, we use a pretrained model for these features.
Run:

```python preprocess/generate_description_features.py --dataset ImageNet```

This will save the tuple of description strings, 
description features in `cache/description_features__ViT-B-16_openai.tensor`

### (2) Calculate greedy descriptor soups
This needs to be done for each random seed of ImageNet training split! Run:

```python preprocess/get_greedy_descriptor_soup.py --dataset ImageNet --seed 1```

This will save the greedily selected descriptors in `cache/good_descriptions_seed1__ViT-B-16_openai.list` as a list.

## 🍜 Word soups
--------------------

### (1) Get Word Features
`preprocess/words.list` contains 10,000 most common English words minus swear words. They have a space prepended. We can use the same `preprocess/generate_description_features.py` to generate the text features from individual words. For pretrained model, run:

```python preprocess/generate_description_features.py --dataset ImageNet --descriptions preprocess/words.list --savename word_features ```

This will save the tuple or words and word features in `cache/word_features__ViT-B-16_openai.tensor`

### (2) Calculate greedy word soups
This needs to be done for each random seed of ImageNet training split! Run:

```python preprocess/get_greedy_word_soup.py --dataset ImageNet --seed 1 --n_descriptors 250```

This will save the greedily selected descriptors in `cache/word_soup_descriptors_seed1__ViT-B-16_openai.list` as a list.

## 🧪 Baselines
-----------------

### CE

Descriptor soup

```bash
python main_crossdataset.py --save_model 1 --seed 1 \
--descriptor_file cache/good_descriptions_seed1__ViT-B-16_openai.list \
--openai_eval 1 \
--gpt_centroid_eval 1 \
--gpt_score_averaging_eval 1 \
--soup_eval 1 \
--token_offset_eval 1 > ce_1.o
```

word soup

```bash
python main_crossdataset.py --eval_only 1 --seed 1 \
--checkpoint {} \
--descriptor_file cache/word_soup_descriptors_seed1__ViT-B-16_openai.list \
--soup_eval 1 \
--token_offset_eval 1 > ce_1.word_soup.o
```

(New 10/8) combined descriptor soup and word soup evaluation without needing to pass in descriptor list file name:

```bash
python main_crossdataset.py --eval_only 1 --seed 1 \
--use_cached_image_features 1 \
--openai_eval 1 \
--gpt_centroid_eval 1 \
--gpt_score_averaging_eval 1 \
--soup_eval 1 \
--token_offset_eval 1
```

### learning rate tuning

Use `run_ce.sh`, `run_coop.sh`, `run_maple.sh`, and `run_clipood.sh` to tune the learning rate.
They run three seeds. Use `parse.ipynb` to parse the output logs into something you ca paste into a spreadsheet. Example usage:

```bash
sh run_coop.sh 0.0001 ViT-B-32 512
```

current best learning rates:

| Method | ViT-B-32 | ViT-B-16 | ViT-L-14 |
| ------ | -------- | -------- | -------- | 
| CE     | 2e-5 | 2e-5 | 2e-5 |
| Clipood | 5e-6 | 2e-5 | 2e-5 |
| CoOp   | 8e-5 | 8e-5 | 8e-5 |
| MaPLe | 1 | 1 | 1 |

### Evaluation

Once learning rate has been tuned, run the full evaluation with the best learning rate for each,use `*_with_eval.sh` scripts.

## 🧪 Base to novel setting
-----------------------------

First, generate features again:

For descriptor features:

```bash
python generate_description_features.py --dataset ImageNet --subsample_classes base
python generate_description_features.py --dataset Caltech101 --subsample_classes base
python generate_description_features.py --dataset OxfordPets --subsample_classes base
python generate_description_features.py --dataset StanfordCars --subsample_classes base
python generate_description_features.py --dataset Flowers102 --subsample_classes base
python generate_description_features.py --dataset Food101 --subsample_classes base
python generate_description_features.py --dataset FGVCAircraft --subsample_classes base
python generate_description_features.py --dataset SUN397 --subsample_classes base
python generate_description_features.py --dataset DTD --subsample_classes base
python generate_description_features.py --dataset EuroSAT --subsample_classes base
python generate_description_features.py --dataset UCF101 --subsample_classes base
```

For word features:

```bash
python generate_description_features.py --dataset ImageNet --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset Caltech101 --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset OxfordPets --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset StanfordCars --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset Flowers102 --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset Food101 --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset FGVCAircraft --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset SUN397 --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset DTD --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset EuroSAT --descriptions words.list --savename word_features --subsample_classes base
python generate_description_features.py --dataset UCF101 --descriptions words.list --savename word_features --subsample_classes base
```

To get greedy descriptor soup:

```bash
sh run_get_greedy_descriptor_soup.sh ImageNet
sh run_get_greedy_descriptor_soup.sh Caltech101
sh run_get_greedy_descriptor_soup.sh OxfordPets
sh run_get_greedy_descriptor_soup.sh StanfordCars
sh run_get_greedy_descriptor_soup.sh Flowers102
sh run_get_greedy_descriptor_soup.sh Food101
sh run_get_greedy_descriptor_soup.sh FGVCAircraft
sh run_get_greedy_descriptor_soup.sh SUN397
sh run_get_greedy_descriptor_soup.sh DTD
sh run_get_greedy_descriptor_soup.sh EuroSAT
sh run_get_greedy_descriptor_soup.sh UCF101
```

To get greedy word soup:

```bash
sh run_get_greedy_word_soup.sh ImageNet
sh run_get_greedy_word_soup.sh Caltech101
sh run_get_greedy_word_soup.sh OxfordPets
sh run_get_greedy_word_soup.sh StanfordCars
sh run_get_greedy_word_soup.sh Flowers102
sh run_get_greedy_word_soup.sh Food101
sh run_get_greedy_word_soup.sh FGVCAircraft
sh run_get_greedy_word_soup.sh SUN397
sh run_get_greedy_word_soup.sh DTD
sh run_get_greedy_word_soup.sh EuroSAT
sh run_get_greedy_word_soup.sh UCF101
```

Then run training using provided bash scripts, example:

```sh run_ce_with_eval.btn.sh 5e-05 > run_ce_with_eval.btn.sh_5e-05.o ```

## CoOp soft descriptor ensemble baseline
----------------------------------------------------------

Run `coop_soft_descriptor_ensemble.sh` which logs in `train_softd.o` and outputs 

* `cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed1.list_e0.soft`
* `cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed2.list_e0.soft`
* `cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed3.list_e0.soft`

These are list of 8 soft descriptors.

***To evaluate***:
(reference `run_soft.sh`) 

## More baselines
--------------------------

Many more baselines in the `scripts/` folder. Run these at your pleasure.
