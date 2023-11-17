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

import source.gpt_helpers as gpt_helpers

from utils import *
from losses import *
from samplers import *
from transforms import *
from models import *
from trainer import *
from argparse_parameters import get_arg_parser
import time

parser = get_arg_parser()
parser.add_argument('--descriptor', default = '', type=str)
parser.add_argument('--descriptor_file', default = '', type=str)
args = parser.parse_args()
print(args)

assert not len(args.descriptor) > 0 and len(args.descriptor_file) > 0

if len(args.descriptor_file) > 0:
    descriptors = torch.load(args.descriptor_file)
else:
    descriptors = [args.descriptor]

n_classes = 1000
d = args.d
tokenizer = open_clip.get_tokenizer(args.modelname)

####################################################################
####################################################################
####################################################################

def _evaluate(image_features, text_features, y_truth):
    with torch.no_grad():
        assert image_features.shape[0] == y_truth.shape[0]
        y_truth = y_truth.cuda()
        probs = image_features @ F.normalize(text_features).T
        y_hat = probs.max(1).indices
    acc = (y_hat == y_truth).float().mean().item() * 100.
#     print('acc: ', acc)
    return acc

dataset = args.dataset

cfg = argparse.Namespace()
cfg.ROOT = args.data_dir
cfg.NUM_SHOTS = 16
cfg.SEED = 1
cfg.SUBSAMPLE_CLASSES = 'all'
dataset_class = dataset_classes[dataset]
dset = dataset_class(cfg)
test_xform = get_test_transform()

fn = 'cache/vit_b_16/image_features.y_truth.{}{}.tup'.format(
                dataset, '')
image_features, y_truth = torch.load(fn)
image_features = F.normalize(image_features.float().cuda())
print('loaded image features from {}'.format(fn))

prompt = prompt_strings[dataset] # used this for the scatter plots
# prompt = 'a photo of a {}.' # used this for the tables
text = tokenizer(get_text_labels(dset.classnames, prompt))
n_classes = len(dset.classnames)

model =  MyClip(args.modelname, args.pretrained, n_classes, d, 
                  temp = args.temp, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text,
                cache_dir=args.cache_dir, descriptors=None)
model = model.cuda()
model.eval()

string_texts = get_text_labels(dset.classnames, prompt)
for si in string_texts:
    assert si[-1] == '.'
    
accs = []
for descriptor in tqdm(descriptors):
    string_texts_w_desc = [si[:-1] + ',' + descriptor for si in string_texts]
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        text_features = F.normalize(model.encode_text(tokenizer(string_texts_w_desc).cuda()).float())
    _acc = _evaluate(image_features, text_features, y_truth)
    accs.append(_acc)
torch.save(accs, '{}.{}.accs'.format(args.descriptor_file, dataset))

# if args.descriptor != '':
#     string_texts = [si[:-1] + ',' + args.descriptor for si in string_texts]
# print(string_texts)

############# DIDN'T END UP USING THIS: #############
# f_centroids = torch.zeros((n_classes, args.d), device='cuda')
# for c in range(n_classes):
#     f_centroids[c,:] = image_features[y_truth == c].mean(0)
# f_centroids = F.normalize(f_centroids.float())

# uniformity = multimodal_uniformity(f_centroids, text_features).item()
# alignment = alignment_loss(f_centroids, text_features).item()

############# FOLLOWING IS WHAT I ENDED UP USING FOR THE TABLES: #############
# alignment = alignment_loss(image_features, text_features[y_truth]).item()
# uniformity = multimodal_uniformity(image_features, text_features).item()
# print('(alignment, uniformity):', (alignment, uniformity))
















