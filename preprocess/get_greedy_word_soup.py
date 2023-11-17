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
parser.add_argument('--k0', default = 250, type =int)
parser.add_argument('--always_select_best', default = 0, type =int) # overrides k0
parser.add_argument('--k1', default = 1000, type =int)
parser.add_argument('--patience', default = 250, type =int)
parser.add_argument('--max_word_length', default = 10, type =int)
args = parser.parse_args()
print(args)

seed = args.seed
checkpoint = args.checkpoint
first_threshold = args.k0
second_threshold = args.k1
n_descriptors = args.n_descriptors

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
load_path = 'cache/{}word_features_{}_{}_{}.tensor'.format(
    '' if args.subsample_classes == 'all' else dataset,
    '', ### HACK
    modelname,
    pretrained
)
words, word_features = torch.load(
    load_path
)
for word in words:
    assert word[0] == ' '
words = [word[1:] for word in words]
print('loading features from {}.'.format(load_path))

assert len(word_features) == len(words)
print('word_features.shape: ', word_features.shape)

print('calculating individual accuracies of words')
accs = []
for i in tqdm(range(len(words))):
    acc = _get_acc(image_features, word_features[i], y_truth)
    accs.append(acc)
del word_features
sorted_indices = torch.tensor(accs).sort(descending=True).indices

words_new = []
for ind in sorted_indices:
    words_new.append(words[ind])
words = words_new

print('max individual word acc: ', max(accs))
print('best word: ')
print(words[0])

good_descriptions = []
for index in range(n_descriptors):
    running_accs = [0.0]
    prompted_strings = get_text_labels(base_dset.classnames, prompt)
    prompted_strings = [si[:-1] + ',' for si in prompted_strings]
    counter = 0

    # pool of random good words
    selected_words = deepcopy(words[:second_threshold])
    random.shuffle(selected_words)
    selected_words = selected_words[:args.patience]
    
    # randomly select a good first word
    first_word_index = torch.randperm(first_threshold)[0].item()
    if args.always_select_best:
        first_word_index = index
    first_word = words[first_word_index]
    print('first_word: ', first_word)
    
    for word in tqdm([first_word] + selected_words):
        next_prompt_strings = [si + ' ' + word for si in prompted_strings]
        model.reset_text(tokenizer(next_prompt_strings))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            f_text = F.normalize(
                model.get_text_prototypes()
            )
        acc = _get_acc(image_features, f_text, y_truth)
        if acc > running_accs[-1]:
            running_accs.append(acc)
            prompted_strings = next_prompt_strings
            counter += 1
            print('acc now {}, example: {}'.format(running_accs[-1], prompted_strings[0]))

        if counter >= args.max_word_length:
            break

    good_desc = ' ' + prompted_strings[0].split(',')[-1].strip()
    good_descriptions.append(good_desc)
    print('Final word chain acc: {}'.format(running_accs[-1]))

good_descriptions = [di+'.' for di in good_descriptions]
torch.save(
    good_descriptions, 
    'cache/{}word_soup_descriptors_seed{}_{}_{}_{}.list'.format(
        '' if args.subsample_classes == 'all' else dataset,
        args.seed, 
        '' if checkpoint == '' else checkpoint.split('/')[-1], 
        modelname,
        pretrained
    )
)


