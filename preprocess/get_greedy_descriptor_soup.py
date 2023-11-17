import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))

import argparse
import os
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import open_clip
from copy import deepcopy
from source.utils import *
from source.losses import *
from source.samplers import *
from source.transforms import *
from source.models import *
from source.trainer import *
from argparse_parameters import get_arg_parser
import sys

parser = get_arg_parser()
parser.add_argument('--subsample_classes', default = 'all', type=str)
args = parser.parse_args()
print(args)

seed = args.seed
checkpoint = args.checkpoint

base_cfg = argparse.Namespace()
base_cfg.ROOT = args.data_dir
base_cfg.NUM_SHOTS = 16
base_cfg.SEED = args.seed
base_cfg.SUBSAMPLE_CLASSES = args.subsample_classes
device = "cuda"
bs = args.bs
modelname = args.modelname
pretrained = args.pretrained
cache_dir = args.cache_dir
d = args.d
epochs = args.n_epochs
iters_per_epoch = args.iters_per_epoch
# assert args.dataset == 'ImageNet' # source dataset
dataset = args.dataset

prompt = prompt_strings[args.dataset]
dataset_class = dataset_classes[args.dataset]

base_dset = dataset_class(base_cfg)
test_xform = get_test_transform()
dset_base_train = dassl_dataset_conversion(base_dset, test_xform, 'train')
dl_base_train = torch.utils.data.DataLoader(
                dset_base_train,
                num_workers=8,
                batch_size=bs,
                pin_memory=True,
                shuffle=False
                        )
print('len(dset_base_train): ', len(dset_base_train))
print('number of base classes: ', len(base_dset.classnames))

n_classes = len(base_dset.classnames)
tokenizer = open_clip.get_tokenizer(modelname)
text_base = tokenizer(get_text_labels(base_dset.classnames, prompt))

model =  MyClip(modelname, pretrained, n_classes, args.d, 
                  temp = args.temp, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text_base,
                cache_dir=cache_dir)
model = model.cuda()

if checkpoint != '':
    print('LOADING CHECKPOINT {}'.format(checkpoint))
    sd = torch.load(checkpoint)
    model.load_state_dict(sd)

image_features, y_truth = get_features(dl_base_train, model, d=args.d)
image_features = F.normalize(image_features.cuda())
y_truth = y_truth.cuda()

def _get_acc(image_features, f_text, y_truth):
    scores = image_features @ f_text.cuda().T
    y_hat = scores.max(1).indices
    acc = (y_hat == y_truth).float().mean().item() * 100.
    return acc

######################################################
load_path = 'cache/{}description_features_{}_{}_{}.tensor'.format(
    '' if args.subsample_classes == 'all' else dataset,
    checkpoint, 
    modelname,
    pretrained
)

descriptions, description_features = torch.load(
    load_path
)
print('loading features from {}.'.format(load_path))

assert len(description_features) == len(descriptions)
print('description_features.shape: ', description_features.shape)

print('calculating individual accuracies of descriptors')
accs = []
for i in tqdm(range(len(descriptions))):
    acc = _get_acc(image_features, description_features[i], y_truth)
    accs.append(acc)

sorted_indices = torch.tensor(accs).sort(descending=True).indices
description_features = description_features[sorted_indices]
descriptions_new = []
for ind in sorted_indices:
    descriptions_new.append(descriptions[ind])
descriptions = descriptions_new

print('max individual acc: ', max(accs))
print('best descriptor: ')
print(descriptions[0])

print('calculating greedy soup starting with best descriptor')
accs = [_get_acc(image_features, description_features[0], y_truth)]
good_descriptions = [descriptions[0]]
running_text_feature_list = [description_features[0]]

for i in tqdm(range(1, len(descriptions))):
    description = descriptions[i]
    description_feature = description_features[i]
    
    # try adding to soup
    running_text_feature_list.append(description_feature)
    f_text = F.normalize(torch.stack(running_text_feature_list).mean(0))
    acc = _get_acc(image_features, f_text, y_truth)

    if acc > accs[-1]:
        # add to soup
        accs.append(acc)
        print('acc now: ', acc, description)
        good_descriptions.append(description)
    else:
        # remove from soup
        running_text_feature_list.pop()

print('number of descriptors in soup: ', len(good_descriptions))
torch.save(
    good_descriptions, 
    'cache/{}good_descriptions_seed{}_{}_{}_{}.list'.format(
        '' if args.subsample_classes == 'all' else dataset,
        args.seed, 
        checkpoint, 
        modelname,
        pretrained
    )
)



