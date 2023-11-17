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
parser.add_argument('--descriptor_file', default = '', type=str)
args = parser.parse_args()
print(args)

# use default descriptor files
if len(args.descriptor_file) > 0 :
    default_word_descriptor_file = args.descriptor_file
else:
    default_word_descriptor_file = 'cache/word_soup_descriptors_seed{}__{}_{}.list'.format(args.seed, args.modelname, args.pretrained)
word_descriptors = torch.load(default_word_descriptor_file)
    # word_descriptors = [' conventional contribute exists purposes interest along term favourite wikipedia.']

print('CHOSEN WORD SOUP DESCRIPTORS: ')
for desc in word_descriptors:
    print(desc)
        
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
        
### Get the zeroshot text prototypes (with prompt)
tokenizer = open_clip.get_tokenizer(modelname)
text_base = tokenizer(get_text_labels(base_dset.classnames, prompt))

############################ TRAINING LOOP #########################

sd_list = []
for epoch in range(args.n_epochs):
    sd_list.append([])
    
for descriptor in word_descriptors:
    model =  MyClip(modelname, pretrained, n_classes, d, 
                  temp = args.temp, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text_base,
                cache_dir=cache_dir, descriptors=[])
    model = model.cuda()
    print('training descriptor: {}'.format(descriptor))
    model.shallow_prompt.swap_suffix(descriptor, model)
    ema_model = deepcopy(model)
    
    params = get_params(args, model)
    optimizer, scheduler, scaler = get_optimizer(args, params)
    
    train_it = ForeverDataIterator(dl_base_train)
    
    save_path = 'cache/soft_descriptors/{}.soft'.format(default_word_descriptor_file.split('/')[-1])
    init_sd = deepcopy(ema_model.shallow_prompt.state_dict())
    torch.save(init_sd, save_path)
    
    for epoch in range(args.n_epochs):
        print('epoch: {}'.format(epoch))
        model, ema_model = train_loop(
                args, model, ema_model, train_it, 
                n_classes, None,
                scaler, optimizer, scheduler,
                tokenizer=tokenizer
            )
        ema_model.eval()
        ema_model.cpu()
        sd_list[epoch].append(deepcopy(ema_model.shallow_prompt.state_dict()))
        ema_model.cuda()
        
    del model, ema_model
        
for epoch in range(args.n_epochs):
    save_path = 'cache/soft_descriptors/random_8_10_token_8_ensemble/{}_e{}.soft'.format(
        default_word_descriptor_file.split('/')[-1], epoch)
    torch.save(sd_list[epoch], save_path)
        
        
        
        