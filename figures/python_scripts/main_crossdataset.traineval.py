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

experiment_startime = int(time.time() * 1000000)

parser = get_arg_parser()
parser.add_argument('--subsample_classes', default = 'all', type=str)
parser.add_argument('--descriptor_file', default = '', type=str)
parser.add_argument('--shuffle_descriptors', default = 0, type=int)
parser.add_argument('--use_cached_image_features', default = 0, type=int)
parser.add_argument('--use_pretrained_image_features', default = 0, type=int)
parser.add_argument('--openai_eval', default = 0, type=int)
parser.add_argument('--gpt_centroid_eval', default = 0, type=int)
parser.add_argument('--gpt_score_averaging_eval', default = 0, type=int)
parser.add_argument('--soup_eval', default = 0, type=int)
parser.add_argument('--score_averaging', default = 0, type=int)
parser.add_argument('--token_offset_eval', default = 0, type=int)
parser.add_argument('--zs_average', default = 0, type=int)
parser.add_argument('--rand_seed', default = 0, type=int)
args = parser.parse_args()
print(args)

descriptors = []
word_descriptors = None
good_descriptors = None

if args.descriptor_file != '':
    descriptors = torch.load(args.descriptor_file)
    if args.n_descriptors > 0:
        if args.shuffle_descriptors:
            random.shuffle(descriptors)
        descriptors = descriptors[:args.n_descriptors]
        
    print('CHOSEN DESCRIPTORS: ')
    for desc in descriptors:
        print(desc)
else:
    # use default descriptor files
    default_word_descriptor_file = 'cache/word_soup_descriptors_seed{}__{}_{}.list'.format(args.seed, args.modelname, args.pretrained)
    default_desc_descriptor_file = 'cache/good_descriptions_seed{}__{}_{}.list'.format(args.seed, args.modelname, args.pretrained)
    
    word_descriptors = torch.load(default_word_descriptor_file)
    good_descriptors = torch.load(default_desc_descriptor_file)
    
    print('CHOSEN WORD SOUP DESCRIPTORS: ')
    for desc in word_descriptors:
        print(desc)
        
    print('CHOSEN DESC SOUP DESCRIPTORS: ')
    for desc in good_descriptors:
        print(desc)

if args.rand_seed > 0:
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed) # set random seed for all gpus
    
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

dset_base_train = dassl_dataset_conversion(base_dset, test_xform, 'train')

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

dl_base_train = torch.utils.data.DataLoader(
                dset_base_train,
                num_workers=8,
                batch_size=128,
                pin_memory=True,
                shuffle=False, drop_last=False,
            )
####################################################################

############################ MODEL #################################
n_classes = len(base_dset.classnames)
        
### Get the zeroshot text prototypes (with prompt)
tokenizer = open_clip.get_tokenizer(modelname)
text_base = tokenizer(get_text_labels(base_dset.classnames, prompt))

model =  MyClip(modelname, pretrained, n_classes, d, 
                  temp = args.temp, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text_base,
                cache_dir=cache_dir, descriptors=descriptors)
model = model.cuda()

init_sd = deepcopy(model.cpu().state_dict())
model = model.cuda()

# if maple is true or deep visual and text prompts,
# need to not include that in the initial frozen model
if args.maple or args.visual_prompt_depth or args.text_prompt_depth > 1:
    init_args = deepcopy(args)
    init_args.maple = 0
    init_args.visual_prompt_depth = 0
    init_args.text_prompt_depth = 1
    init_frozen_model =  MyClip(modelname, pretrained, n_classes, d, 
                  temp = args.temp, args=init_args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text_base,
                cache_dir=cache_dir, descriptors=descriptors)
    init_frozen_model = init_frozen_model.cuda()
else:
    init_frozen_model = deepcopy(model)
init_frozen_model.eval()
####################################################################

# model_path = 'cache/crossdataset_prompt_sd_1697515571017542_e10.pt'

def _evaluate(image_features, text_features, y_truth):
    with torch.no_grad():
        assert image_features.shape[0] == y_truth.shape[0]
        y_truth = y_truth.cuda()
        image_features = F.normalize(image_features.float().cuda())
        probs = image_features @ F.normalize(text_features).T
        y_hat = probs.max(1).indices
    acc = (y_hat == y_truth).float().mean().item() * 100.
    print('acc: ', acc)
    return acc

def _get_average_kl(
    model,
                image_features,
                text_features_list,
                y_truth
            ):
    def _kl(p,q):
        return (p * (p + 1e-8).log() - p * (q + 1e-8).log()).sum(1).mean()
    kls = []  
    for i in range(len(text_features_list) - 1):
        with torch.no_grad():
            assert image_features.shape[0] == y_truth.shape[0]
            y_truth = y_truth.cuda()
            image_features = F.normalize(image_features.float().cuda())
            probs1 = (model.temp * image_features @ F.normalize(text_features_list[i]).T).softmax(1)
            probs2 = (model.temp * image_features @ F.normalize(text_features_list[i+1]).T).softmax(1)
            kls.append((_kl(probs1, probs2) + _kl(probs2, probs1)) * 0.5)

    return (sum(kls) / float(len(kls))).item()
    
train_accs = []
average_kl_list = []
image_features = None

for epoch in range(args.n_epochs):
    print('epoch {}'.format(epoch))
    
    model_path = args.checkpoint.format(epoch)
    print('evaluating {}'.format(model_path))
    model.shallow_prompt.load_state_dict(torch.load(model_path))
    
    # reset descriptors and class texts
    model.shallow_prompt.reset_descriptors(model, word_descriptors)
    model.reset_text(text_base)
    model.eval()
    model.cuda()
    
    with torch.no_grad():
        if image_features is None:
            # assume image encoder not trained !!!
            image_features, y_truth = get_features(dl_base_train, model, d=args.d)
#         text_features = F.normalize(model.get_text_prototypes().float())

        # make feature list
        text_features_list = []
        for descriptor_index in range(len(model.shallow_prompt.desc_vectors)):
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                text_features_sub = model.get_text_prototypes(
                    descriptor_index=descriptor_index,
                    token_offset=0
                )
            text_features_list.append(F.normalize(text_features_sub.float()))
        
        average_kl_list.append(
            _get_average_kl(
                model,
                image_features,
                text_features_list,
                y_truth
            )
        )
        train_accs.append(
            _evaluate(
                image_features,
                F.normalize(torch.stack(text_features_list).mean(0)), 
                y_truth
            )
        )
    
print(train_accs)
print(average_kl_list)

# scripts/run_coop_regularized_detailed.sh

# junk/1018/run_coop_regularized_output_lam_0.0_temp_10_detailed.o
# seed 1: cache/crossdataset_prompt_sd_1697515571017542_e{}.pt
# seed 2: cache/crossdataset_prompt_sd_1697521556144958_e{}.pt
# seed 3: cache/crossdataset_prompt_sd_1697527725632252_e{}.pt

# junk/1018/run_coop_regularized_output_lam_0.1_temp_10_detailed.o
# seed 1: cache/crossdataset_prompt_sd_1697509303813660_e{}.pt
# seed 2: cache/crossdataset_prompt_sd_1697515243275625_e{}.pt
# seed 3: cache/crossdataset_prompt_sd_1697521307267964_e{}.pt

# junk/1018/run_coop_regularized_output_lam_0.25_temp_10_detailed.o
# seed 1: cache/crossdataset_prompt_sd_1697504798152194_e{}.pt
# seed 2: cache/crossdataset_prompt_sd_1697513125765729_e{}.pt
# seed 3: cache/crossdataset_prompt_sd_1697521529700174_e{}.pt

















