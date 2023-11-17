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

from source.utils import *
from source.losses import *
from source.samplers import *
from source.transforms import *
from source.models import *
from source.trainer import *
from argparse_parameters import get_arg_parser

parser = get_arg_parser()
parser.add_argument('--subsample_classes', default = 'all', type=str)
parser.add_argument('--descriptor_file', 
                    default = '', type=str)
args = parser.parse_args()
print(args)

# descriptors = torch.load('cache/good_descriptions_seed{}.list'.format(
#         args.seed))

if args.descriptor_file != '':
    descriptors_sd_list = torch.load(args.descriptor_file.format(args.seed))
    if not type(descriptors_sd_list) is list:
        descriptors_sd_list = [descriptors_sd_list]
    suffixes = []
    for d in descriptors_sd_list:
        assert len(d) == 1
        assert 'suffix_vectors' in d
        suffixes.append(d['suffix_vectors'])
    
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
        
    # for proDA evaluation
    assert 'proda_descriptor_vectors' in sd
    suffixes = []
    for i in range(len(sd['proda_descriptor_vectors'])):
        suffixes.append(sd['proda_descriptor_vectors'][i,:,:])
    del sd['proda_descriptor_vectors']
    
    model.load_state_dict(sd)
    
print('len(suffixes): ', len(suffixes))

############################ OPTIMIZERS ############################
ema_model = deepcopy(model)
####################################################################

ema_model.eval()
model_copy = deepcopy(ema_model)

########################################################
print('Runing Evaluation ...')
model_copy.cuda()
model_copy.eval()

return_dict = {}
dataset_list = list(dataset_classes.keys())
tim_accum = 0.
accs = []
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

    print('{} has {} test samples'.format(dataset, len(dset_test)))

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
    fn = 'cache/image_features.y_truth.{}{}.tup'.format(
        dataset, '')

    image_features, y_truth = torch.load(fn)
    print('loaded image features from {}'.format(fn))
    ##################################

    def _evaluate(image_features, text_features, y_truth):
        with torch.no_grad():
            assert image_features.shape[0] == y_truth.shape[0]
            y_truth = y_truth.cuda()
            image_features = F.normalize(image_features.float().cuda())
            probs = image_features @ F.normalize(text_features).T
            y_hat = probs.max(1).indices
        acc = (y_hat == y_truth).float().mean().item() * 100.
#         print('acc: ', acc)
        return acc

#         ### a photo of ZS
    model_copy.reset_text(text)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        text_features = model_copy.get_text_prototypes()

    print('a photo of ZS')
    acc1 = _evaluate(image_features, text_features, y_truth)

    def _get_descriptor_feature_matrix( model_copy, suffixes, text):
        model_copy.reset_text(text)
        text_features_list = []
        for suffix in suffixes:
            model_copy.shallow_prompt.reset_suffix_vectors(suffix)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                text_features_sub = model_copy.get_text_prototypes()
            text_features_list.append(F.normalize(text_features_sub.float()))
        model_copy.shallow_prompt.reset_suffix_vectors(None)
        return torch.stack(text_features_list)

    def _get_descriptor_centroids( 
        model_copy,suffixes,
        text, descriptor_scores=None):
        text_features_stack = _get_descriptor_feature_matrix(
            model_copy,suffixes,text
        )
        if descriptor_scores is None:
            return F.normalize(text_features_stack.mean(0))
        else:
            _descriptor_scores = descriptor_scores.repeat(1)
            _descriptor_scores = _descriptor_scores / _descriptor_scores.sum()
            return F.normalize(
                (_descriptor_scores[:,None,None] * text_features_stack).sum(0)
            )

    import time
    start = time.time()
    text_features = _get_descriptor_centroids(
        model_copy, suffixes, text,
        descriptor_scores=None)
    end = time.time()
    tim_accum += end-start
    acc5 = _evaluate(image_features, text_features, y_truth)
    print('no token offset smart prompt choices ensemble')
    print('acc: ', acc5)
    
    ### score averaging 
    text_features_stack = _get_descriptor_feature_matrix(
            model_copy,suffixes,text
        )
    
    for i in range(len(text_features_stack)):
        __acc = _evaluate(image_features, text_features_stack[i,:,:], y_truth)
        print('descriptor #{} acc: {}'.format(i, __acc))
    
    scores = torch.zeros(image_features.shape[0], len(dset.classnames)).cuda()
    for i, c in enumerate(dset.classnames):
        v = F.normalize(text_features_stack[:, i, :])
        scores[:, i] = (image_features.float().cuda() @ v.T).mean(dim=1)
    acc7 = (scores.max(1).indices == y_truth.cuda()).float().mean().item() * 100.
    print(' +++ score averaging')
    print('acc: ', acc7)

    print(accs)
    return_dict[dataset] = [acc1, acc5, acc7]

print('Results:')
print('', end=',')
for dataset in dataset_list:
    print(dataset, end=',')
print()
metrics = [
            'ZS', #'GPT', 'GPT-score-mean',
            'word-soup',
            'score-average'
          ]
for i, metric in enumerate(metrics):
    print(metric, end=',')
    for dataset in dataset_list:
        print(return_dict[dataset][i], end=',')
    print()

print('text features time: ', tim_accum)