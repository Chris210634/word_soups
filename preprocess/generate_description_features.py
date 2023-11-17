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

parser = get_arg_parser()
parser.add_argument('--subsample_classes', default = 'all', type=str)
parser.add_argument('--descriptions', default = 'preprocess/descriptions.list', type =str)
parser.add_argument('--savename', default = 'description_features', type =str)
args = parser.parse_args()
print(args)

checkpoint = args.checkpoint

dataset    = args.dataset
modelname  = args.modelname
pretrained = args.pretrained
data_dir   = args.data_dir
cache_dir  = args.cache_dir
prompt = prompt_strings[dataset]

cfg = argparse.Namespace()
cfg.ROOT = data_dir
cfg.NUM_SHOTS = 16
cfg.SEED = 1
cfg.SUBSAMPLE_CLASSES = args.subsample_classes
dataset_class = dataset_classes[dataset]
dset = dataset_class(cfg)
n_classes = len(dset.classnames)
        
### Get the zeroshot text prototypes (with prompt)
tokenizer = open_clip.get_tokenizer(modelname)
text_base = tokenizer(get_text_labels(dset.classnames, prompt))

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

tokenizer = open_clip.get_tokenizer(modelname)
prompted_strings = get_text_labels(dset.classnames, prompt)
assert all([c[-1] == '.' for c in prompted_strings])

descriptions = torch.load(args.descriptions)
description_features = torch.zeros(
    len(descriptions), 
    len(dset.classnames), 
    args.d
)

for i, desc in enumerate(tqdm(descriptions)):
    dl = [c[:-1] + ',' + desc for c in prompted_strings]
    
    # print one out for sanity check
    if i == 0:
        print('Example: ', dl[0])
        
    model.reset_text(tokenizer(dl))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        f_text = F.normalize(
            model.get_text_prototypes()
        )
    description_features[i, :, :] = f_text
    
torch.save(
    (descriptions,description_features), 
    'cache/{}{}_{}_{}_{}.tensor'.format(
        '' if args.subsample_classes == 'all' else dataset,
        args.savename,
        checkpoint, 
        modelname,
        pretrained
    )
)
