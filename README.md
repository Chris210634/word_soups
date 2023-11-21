# Word and Descriptor Soups 🍜
-----------------------------------------------------

![](https://github.com/Chris210634/word_soups/blob/main/figures/parameter_efficiency.png?raw=true)

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

**Run:** `python preprocess/generate_description_features.py --dataset ImageNet`

This will save the tuple of description strings, 
description features in `cache/description_features__ViT-B-16_openai.tensor`

### (2) Calculate greedy descriptor soups
This needs to be done for each random seed of ImageNet training split! 

**Run:** 

```bash
python preprocess/get_greedy_descriptor_soup.py --dataset ImageNet --seed 1
python preprocess/get_greedy_descriptor_soup.py --dataset ImageNet --seed 2
python preprocess/get_greedy_descriptor_soup.py --dataset ImageNet --seed 3
```

This will save the greedily selected descriptors in `cache/good_descriptions_seed1__ViT-B-16_openai.list` as a list.

## 🍜 Word soups
--------------------

### (1) Get Word Features
`preprocess/words.list` contains 10,000 most common English words minus swear words. They have a space prepended. We can use the same `preprocess/generate_description_features.py` to generate the text features from individual words.

**Run:** ```python preprocess/generate_description_features.py --dataset ImageNet --descriptions preprocess/words.list --savename word_features ```

This will save the tuple or words and word features in `cache/word_features__ViT-B-16_openai.tensor`

### (2) Calculate greedy word soups
This needs to be done for each random seed of ImageNet training split!

**Run:**

```bash
python preprocess/get_greedy_word_soup.py --dataset ImageNet --seed 1 --n_descriptors 250
python preprocess/get_greedy_word_soup.py --dataset ImageNet --seed 2 --n_descriptors 250
python preprocess/get_greedy_word_soup.py --dataset ImageNet --seed 3 --n_descriptors 250
```

This will save the greedily selected descriptors in `cache/word_soup_descriptors_seed1__ViT-B-16_openai.list` as a list.

## 🧪 Baselines
-----------------

Results are outputted in CSV format at the end of the experiment. You can copy and paste directly into a spreadsheet.

### Zero-shot comparisons

For all ZS methods presented in Table 3 of the paper (Open-AI handcrafted ensemble, GPT, descriptor soup, token offest, word soup), run: 

```bash
sh run_pt_eval.sh 0 ViT-B-16 openai 512
```

For WaffleCLIP with 16 members, run:

```bash
sh waffle_descriptors_eval.sh 16
```

### Few-shot OOD comparisons

These scripts train on 3 random splits of 16-shot ImageNet-1K. **"XD Mean"** stands for average test accuracy on 10 OOD ddatasets. **"DG Mean"** stands for average test accuracy on 4 domain-shifted versions of ImageNet. You can verify these results by running the indicated bash script and pasting the CSV-formatted results at the end of the output into a spreadsheet.

| Method | Command to run | XD Mean | DG Mean |
| ------ | -------------- | ------ | ------ |
| CLIP-adapter | `scripts/run_adapter.sh 6e-3 ViT-B-16 512` | 65.02 | 58.12 |
| bitfit | `scripts/bitfit.sh 1.25e-4 ViT-B-16 512` | 66.05 | 59.12 |
| Cross Entropy | `scripts/run_ce.sh 2e-5 ViT-B-16 512` | 66.80 | 60.39 |
| Cross Entropy + word soup + diversity loss | `scripts/run_ce_regularized.sh 0.25 10` | 67.43 | 61.32 |
| ClipOOD | `scripts/run_clipood.sh 2e-5 ViT-B-16 512` | 66.50 | 60.47 |
| ClipOOD + word soup + diversity loss | `scripts/run_clipood_regularized.sh 0.25 10` | 67.42 | 61.23 |
| CoOp | `scripts/run_coop.sh 8e-5 ViT-B-16 512` | 66.52 | 59.25 |
| CoOp + word soup + diversity loss | `scripts/run_coop_regularized.sh 0.25 10` | 67.30 | 60.25 |
| KgCoOp |  `scripts/run_kgcoop.sh 4e-5 ViT-B-16 512` | 66.16 | 58.64 |
| LoRA |  `scripts/run_lora.sh 1e-5 ViT-B-16 512` | 66.19 | 57.93 |
| MaPLe |  `scripts/run_maple.sh 0.025 ViT-B-16 512` | 66.44 | 59.32 |
| MaPLe + word soup + diversity loss |  `scripts/run_maple_regularized.sh` | 66.65 | 60.20 |
| ProDA |  `scripts/run_proda.sh 3.2e-4 ViT-B-16 512` | 66.23 | 58.83 |
| ProGrad |  `scripts/run_prograd.sh 1.28e-3 ViT-B-16 512` | 66.48 | 58.96 |
| ResBlock-adapter | `scripts/run_resblock_adapter.sh 2.5e-3 ViT-B-16 512` | 65.55 | 59.48 |
| SSF | `scripts/run_ssf.sh 1e-4 ViT-B-16 512` | 65.86 | 58.44 |
| VPT | `scripts/run_vpt_deep.sh 0.8 ViT-B-16 512` | 65.16 | 58.42 |

## 🧪 More experiments
-----------------------------

### Base to novel setting

First, generate features for each training dataset:

For descriptor features:

```bash
for dataset in ImageNet Caltech101 OxfordPets StanfordCars Flowers102 Food101 FGVCAircraft SUN397 DTD EuroSAT UCF101;
do
  python preprocess/generate_description_features.py --dataset $dataset --subsample_classes base
done
```

For word features:

```bash
for dataset in ImageNet Caltech101 OxfordPets StanfordCars Flowers102 Food101 FGVCAircraft SUN397 DTD EuroSAT UCF101;
do
  python preprocess/generate_description_features.py --dataset $dataset --descriptions words.list --savename word_features --subsample_classes base
done
```

To get greedy descriptor soup:

```bash
for dataset in ImageNet Caltech101 OxfordPets StanfordCars Flowers102 Food101 FGVCAircraft SUN397 DTD EuroSAT UCF101;
do
  sh scripts/ablations/run_get_greedy_descriptor_soup.sh $dataset
done
```

To get greedy word soup:

```bash
for dataset in ImageNet Caltech101 OxfordPets StanfordCars Flowers102 Food101 FGVCAircraft SUN397 DTD EuroSAT UCF101;
do
  sh scripts/ablations/run_get_greedy_word_soup.sh $dataset
done
```

Then run training using provided bash scripts, example:

```sh scripts/run_ce_with_eval.btn.sh 5e-05 > run_ce_with_eval.btn.sh_5e-05.o ```

See any bash script called `scripts/*.btn.sh`.

### CoOp soft descriptor ensemble baseline

Run `scripts/ablations/coop_soft_descriptor_ensemble.sh` which logs in `train_softd.o` and outputs 

* `cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed1.list_e0.soft`
* `cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed2.list_e0.soft`
* `cache/soft_descriptors/random_8_10_token_8_ensemble/8_random_10_token_word_chains_seed3.list_e0.soft`

These are list of 8 soft descriptors.

***To evaluate***:
(reference `scripts/ablations/run_soft.sh`) 

### More baselines

Many more baselines in the `scripts/ablations` folder. Run these at your pleasure.
