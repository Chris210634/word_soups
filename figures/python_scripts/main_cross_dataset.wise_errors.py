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

parser = get_arg_parser()
parser.add_argument('--subsample_classes', default = 'all', type=str)
parser.add_argument('--init_suffix_file', 
                    default = 'cache/soft_descriptors/word_soup_descriptors_seed{}__ViT-B-16_openai.list.soft', type=str)
parser.add_argument('--descriptor_file', 
                    default = 'cache/soft_descriptors/word_soup_descriptors_seed{}__ViT-B-16_openai.list.soft', type=str)
args = parser.parse_args()
print(args)

# descriptors = torch.load('cache/good_descriptions_seed{}.list'.format(
#         args.seed))

descriptors_sd_list = torch.load(args.descriptor_file.format(args.seed))
if not type(descriptors_sd_list) is list:
    descriptors_sd_list = [descriptors_sd_list]
suffixes = []
for d in descriptors_sd_list:
    assert len(d) == 1
    assert 'suffix_vectors' in d
    suffixes.append(d['suffix_vectors'])
    
descriptors_sd_list = torch.load(args.init_suffix_file.format(args.seed))
if not type(descriptors_sd_list) is list:
    descriptors_sd_list = [descriptors_sd_list]
init_suffixes = []
for d in descriptors_sd_list:
    assert len(d) == 1
    assert 'suffix_vectors' in d
    init_suffixes.append(d['suffix_vectors'])
    
############################ SETUP ############################
base_cfg = argparse.Namespace()
base_cfg.ROOT = args.data_dir
base_cfg.NUM_SHOTS = 16
base_cfg.SEED = args.seed
assert args.subsample_classes == 'all'
base_cfg.SUBSAMPLE_CLASSES = args.subsample_classes
device = "cuda"
bs = args.bs
modelname = args.modelname
pretrained = args.pretrained
cache_dir = args.cache_dir
d = args.d
epochs = args.n_epochs
iters_per_epoch = args.iters_per_epoch
args.dataset = 'ImageNet' # source dataset
dataset = args.dataset

prompt = prompt_strings[args.dataset]
dataset_class = dataset_classes[args.dataset]
####################################################################

############################ TRANSFORMS ############################
train_xform = get_train_transform()
test_xform = get_test_transform()
base_dset = dataset_class(base_cfg)

dset_base_train = dassl_dataset_conversion(base_dset, train_xform, 'train')

# unlike the other datasets, when using immagenet,
# following standard procedure, we use the 50,000 validation images
# for testing
dset_base_val = []
dset_base_test, dset_new_test = get_imagenet_val_dataset(
    test_xform,
    imagenet_root = os.path.join(args.data_dir, 'imagenet'))
for i in range(500):
    assert base_dset.classnames[i] == get_imagenet_classnames()[i]
    
print('len(dset_base_train): ', len(dset_base_train))
print('number of base classes: ', len(base_dset.classnames))
####################################################################
     
############################ DATALOADERS ###########################
fastsampler = FastRandomSampler(
    dset_base_train, 
    batch_size=bs,
    num_iters=iters_per_epoch,
    samples_per_class=args.samples_per_class
)
dl_base_train = torch.utils.data.DataLoader(
                dset_base_train,
                num_workers=8,
                batch_size=bs,
                pin_memory=True,
                sampler=fastsampler
            )
####################################################################

############################ MODEL #################################
n_classes = len(base_dset.classnames)
# model.model.visual.patch_dropout = open_clip.transformer.PatchDropout(prob=0.75)
        
### Get the zeroshot text prototypes (with prompt)
tokenizer = open_clip.get_tokenizer(modelname)
text_base = tokenizer(get_text_labels(base_dset.classnames, prompt))

model =  MyClip(modelname, pretrained, n_classes, d, 
                  temp = args.temp, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text_base,
                cache_dir=cache_dir, descriptors=[])
model = model.cuda()

init_sd = deepcopy(model.cpu().state_dict())
model = model.cuda()
init_frozen_model = deepcopy(model)
init_frozen_model.eval()
####################################################################

if args.checkpoint != '':
    sd = torch.load(args.checkpoint)
    if 'shallow_prompt.desc_vectors' in sd:
        del sd['shallow_prompt.desc_vectors']
    model.load_state_dict(sd)

############################ OPTIMIZERS ############################
ema_model = deepcopy(model)
####################################################################

ema_model.eval()
model_copy = deepcopy(ema_model)

########################################################
print('Runing Evaluation ...')
model_copy.cuda()
model_copy.eval()

dataset_list = list(dataset_classes.keys())

accs_master = []
alphas = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
for alpha in alphas:
    accs = []
    suffix = alpha * init_suffixes[0].cuda() + (1. - alpha) * suffixes[0].cuda()
    for dataset in dataset_list:
        cfg = argparse.Namespace()
        cfg.ROOT = args.data_dir
        cfg.NUM_SHOTS = 16
        cfg.SEED = 1
        cfg.SUBSAMPLE_CLASSES = 'all'
        dataset_class = dataset_classes[dataset]
        dset = dataset_class(cfg)
        test_xform = get_test_transform()

        if not dataset in ['ImageNet']:
            dset_test = dassl_dataset_conversion(dset, test_xform, 'test')
        else:
            # unlike the other datasets, when using immagenet,
            # following standard procedure, we use the 50,000 validation images
            # for testing
            dset_test = get_imagenet_val_dataset(
                test_xform,
                imagenet_root = os.path.join(args.data_dir, 'imagenet'),
                split=False)

#         print('{} has {} test samples'.format(dataset, len(dset_test)))

        dl_test = torch.utils.data.DataLoader(
                    dset_test,
                    num_workers=8,
                    batch_size=32,
                    pin_memory=True,
                    drop_last=False,
                    shuffle=False
                )

        prompt = prompt_strings[dataset]
        tokenizer = open_clip.get_tokenizer(args.modelname)
        text = tokenizer(get_text_labels(dset.classnames, prompt))

        ##################################
        ### LOAD CACHED IMAGE FEATURES ###
        assert args.modelname == 'ViT-B-16'
        fn = 'cache/vit_b_16/image_features.y_truth.{}{}.tup'.format(
            dataset, args.checkpoint.split('/')[-1])

        image_features, y_truth = torch.load(fn)
#         print('loaded image features from {}'.format(fn))
        ##################################

        def _evaluate(image_features, text_features, y_truth):
            with torch.no_grad():
                assert image_features.shape[0] == y_truth.shape[0]
                y_truth = y_truth.cuda()
                image_features = F.normalize(image_features.float().cuda())
                probs = image_features @ F.normalize(text_features).T
                y_hat = probs.max(1).indices
            acc = (y_hat == y_truth).float().mean().item() * 100.
            return acc

        model_copy.reset_text(text)
        model_copy.shallow_prompt.reset_suffix_vectors(suffix)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            text_features = F.normalize(model_copy.get_text_prototypes().float())

        __acc = _evaluate(image_features, text_features, y_truth)

        accs.append(__acc)
    print(accs)
    accs_master.append(accs)
print(accs_master)
    
distance = (init_suffixes[0].cuda() - suffixes[0].cuda()).square().sum().sqrt().item()
print('distance: {}'.format(distance))
    
    
    
    
    
    
    
    
    
    
    
    
    