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
import time

experiment_startime = int(time.time() * 1000000)

parser = get_arg_parser()
parser.add_argument('--subsample_classes', default = 'all', type=str)
parser.add_argument('--num_descriptors', default = 32, type=int)
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

#################################################################
#####################    PRODA RAND INIT     ####################
import open_clip
tokenizer = open_clip.get_tokenizer('ViT-B-16')
word_list = torch.load('preprocess/words.list')
word_list = [wi.strip() for wi in word_list]

counter = 0
descriptors = []
while counter < args.num_descriptors:
    import random
    random.shuffle(word_list)
    descriptor = ' ' + ' '.join(word_list[:9]) + '.'

    suffix_tokens = tokenizer(descriptor).view(-1)
    suffix_eof = suffix_tokens.argmax(dim=-1)
    suffix_tokens = suffix_tokens[1:suffix_eof]
#     print('suffix_tokens shape: ', suffix_tokens.shape)
    if suffix_tokens.shape[0] == 10:
        counter += 1
        print('*' + descriptor)
        descriptors.append(descriptor)
        
#################################################################

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

if args.checkpoint != '':
    sd = torch.load(args.checkpoint)
    if 'shallow_prompt.desc_vectors' in sd:
        del sd['shallow_prompt.desc_vectors']
    model.load_state_dict(sd)
    
if args.suffix_string != '':
    model.shallow_prompt.swap_suffix(args.suffix_string, model)

############################ OPTIMIZERS ############################
if not args.eval_only:
    params = get_params(args, model, descriptor_strings=descriptors)
    optimizer, scheduler, scaler = get_optimizer(args, params)
ema_model = deepcopy(model)
####################################################################

lr = args.lr
loss_list = []
accs = []
train_it = ForeverDataIterator(dl_base_train)

tokenized_text_with_descriptors = None
if args.train_with_descriptors:
    class_label_strings = get_text_labels(base_dset.classnames, prompt)
    n_classes = len(class_label_strings)
    for cls in range(n_classes):
        assert class_label_strings[cls][-1] == '.'
    tokenized_text_with_descriptors = []
    for desc in word_descriptors:
        tokenized_dec = tokenizer([ci[:-1] + ',' + desc for ci in class_label_strings])
        tokenized_text_with_descriptors.append(tokenized_dec)
    tokenized_text_with_descriptors = torch.stack(tokenized_text_with_descriptors).cuda()
    print('tokenized_text_with_descriptors.shape: ',
          tokenized_text_with_descriptors.shape)
        
############################ TRAINING LOOP #########################
for epoch in range(epochs):
    if not args.eval_only:
        print('There are {} param groups'.format(len(optimizer.param_groups)))
        for iii, param_group in enumerate(optimizer.param_groups):
            print(
                'epoch:{} , param_group:{}, len: {}, lr:{}'.format(
                    epoch, iii, len(param_group), param_group['lr'] 
                )
            )
        
    ##################### INVOKE training loop ######################
        model, ema_model = train_loop(
            args, model, ema_model, train_it, 
            n_classes, loss_list,
            scaler, optimizer, scheduler,
            tokenizer=tokenizer,
            init_frozen_model=init_frozen_model,
            tokenized_text_with_descriptors=tokenized_text_with_descriptors,
            epoch=epoch
        )
    #################################################################

    ema_model.eval()
    model_copy = deepcopy(ema_model)
    ft_sd = deepcopy(model_copy.cpu().state_dict())
    
    if args.save_model:
        if args.suffix_string != '':
            save_path = 'cache/shallow_prompt_{}.pt'.format('_'.join(args.suffix_string.split(' ')))
            torch.save(model_copy.shallow_prompt.state_dict(), save_path)
        elif args.save_model == 2: # means save shallow prompt state dict only (for coop)
            save_path = 'cache/crossdataset_prompt_sd_{}_e{}.pt'.format(
                experiment_startime, epoch)
            torch.save(model_copy.shallow_prompt.state_dict(), save_path)
        else:
            save_path = 'cache/crossdataset_ft_sd_{}_e{}.pt'.format(
                experiment_startime, epoch)
            torch.save(ft_sd, save_path)
        print('saved ft_sd in {}'.format(save_path))
        
    ########################################################
    print('Runing Evaluation ...')
    model_copy.cuda()
    model_copy.eval()

    return_dict = {}
    dataset_list = list(dataset_classes.keys())
    tim_accum = 0.
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
#         assert args.modelname == 'ViT-B-16'
        
        if not args.use_pretrained_image_features:
            fn = 'cache/image_features.y_truth.{}{}.tup'.format(
                dataset, args.checkpoint.split('/')[-1])
        else: # e.g. coop does not finetune image encoder
            fn = 'cache/image_features.y_truth.{}{}.tup'.format(
                dataset, '')

        if args.use_cached_image_features and fn.split('/')[-1] in os.listdir('cache'):
            try:
                image_features, y_truth = torch.load(fn)
                print('loaded image features from {}'.format(fn))
            except:
                with torch.no_grad():
                    image_features, y_truth = get_features(
                        dl_test, model_copy, d=args.d) 
                torch.save((image_features, y_truth), fn)

        else:
            with torch.no_grad():
                image_features, y_truth = get_features(dl_test, model_copy, d=args.d)
            if args.eval_only:
                torch.save((image_features, y_truth), fn)
        ##################################

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
        
        ##################################
        metrics = []
        accs = []
        ### a photo of ZS
        model_copy.reset_text(text)
        model_copy.shallow_prompt.reset_suffix_vectors(None)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            zs_text_features = model_copy.get_text_prototypes()
           
        print('a photo of ZS')
        acc1 = _evaluate(image_features, zs_text_features, y_truth)
        
        assert 'proda_descriptor_vectors' in model_copy.state_dict()
        suffixes = []
        for i in range(len(model_copy.state_dict()['proda_descriptor_vectors'])):
            suffixes.append(
                model_copy.state_dict()['proda_descriptor_vectors'][i,:,:].detach()
            )
        print('len(suffixes): ', len(suffixes))

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
        
        metrics = [
            'ZS', #'GPT', 'GPT-score-mean',
            'word-soup',
            'score-average'
          ]
        
    print('Results:')
    print('', end=',')
    for dataset in dataset_list:
        print(dataset, end=',')
    print()
    for i, metric in enumerate(metrics):
        print(metric, end=',')
        for dataset in dataset_list:
            print(return_dict[dataset][i], end=',')
        print()
        
    del text
    del model_copy, ft_sd
    