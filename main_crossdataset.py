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
    __descriptors_list = word_descriptors if not word_descriptors is None else descriptors
    for desc in __descriptors_list:
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
        ## Change this back
        fn = 'cache/image_features.y_truth.{}{}{}.tup'.format(dataset, modelname, pretrained)
    
        if fn.split('/')[-1] in os.listdir('cache'):
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
            torch.save((image_features, y_truth), fn)
        
#         if not args.use_pretrained_image_features:
#             fn = 'cache/image_features.y_truth.{}{}.tup'.format(
#                 dataset, args.checkpoint.split('/')[-1])
#         else: # e.g. coop does not finetune image encoder
#             fn = 'cache/image_features.y_truth.{}{}.tup'.format(
#                 dataset, '')

#         if args.use_cached_image_features and fn.split('/')[-1] in os.listdir('cache'):
#             try:
#                 image_features, y_truth = torch.load(fn)
#                 print('loaded image features from {}'.format(fn))
#             except:
#                 with torch.no_grad():
#                     image_features, y_truth = get_features(
#                         dl_test, model_copy, d=args.d) 
#                 torch.save((image_features, y_truth), fn)

#         else:
#             with torch.no_grad():
#                 image_features, y_truth = get_features(dl_test, model_copy, d=args.d)
#             if args.eval_only and args.train_visual_encoder == 0 and args.visual_prompt_depth == 0 and args.modelname == 'ViT-B-16' and args.pretrained == 'openai':
#                 torch.save((image_features, y_truth), fn)
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
        
        def _get_descriptor_feature_matrix( model_copy,descriptors,text,token_offsets):
            model_copy.shallow_prompt.reset_descriptors(model_copy, descriptors)
            model_copy.reset_text(text)
            text_features_list = []
            for token_offset in token_offsets:
                for descriptor_index in range(len(model_copy.shallow_prompt.desc_vectors)):
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                        text_features_sub = model_copy.get_text_prototypes(
                            descriptor_index=descriptor_index,
                            token_offset=token_offset
                        )
                    text_features_list.append(F.normalize(text_features_sub.float()))
            return torch.stack(text_features_list)
        
        def _get_descriptor_centroids( 
            model_copy,descriptors,
            text,token_offsets, descriptor_scores=None):
            text_features_stack = _get_descriptor_feature_matrix(
                model_copy,descriptors,text,token_offsets
            )
            if descriptor_scores is None:
                return F.normalize(text_features_stack.mean(0))
            else:
                _descriptor_scores = descriptor_scores.repeat(len(token_offsets))
                _descriptor_scores = _descriptor_scores / _descriptor_scores.sum()
                return F.normalize(
                    (_descriptor_scores[:,None,None] * text_features_stack).sum(0)
                )
        
        ##################################
        metrics = []
        accs = []
        ### a photo of ZS
        model_copy.reset_text(text)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            zs_text_features = model_copy.get_text_prototypes()
            # uncomment to test encode_text_wrapper
#             text_features_1 = gpt_helpers.encode_text_wrapper(
#                 model_copy, get_text_labels(dset.classnames, prompt), tokenizer)
#             torch.save((text_features, text_features_1), 'tmp.pt')
#             import sys
#             sys.exit(1)
        print('a photo of ZS')
        acc1 = _evaluate(image_features, zs_text_features, y_truth)
        metrics.append('ZS')
        accs.append(acc1)

        ### openai manual ensemble ZS
        if args.openai_eval:
            text_features = gpt_helpers.get_openai_manual_prompt_template_centroids(
                dset.classnames, model_copy, dataset, tokenizer 
                # , n_descriptors=args.n_descriptors # last argument is for ablations only
            )
            print('openai manual prompt ensemble centroid ZS')
            acc2 = _evaluate(image_features, text_features, y_truth)
            metrics.append('ensemble')
            accs.append(acc2)

        ### centroid of gpt descriptions
        if args.gpt_centroid_eval:
            gpt_descriptions = gpt_helpers.get_gpt_descriptions(dataset=dataset)
            text_features = gpt_helpers.get_gpt_centroids(
                gpt_descriptions, dset.classnames, model_copy, tokenizer)
            print('GPT descriptions centroid')
            acc3 = _evaluate(image_features, text_features, y_truth)
            metrics.append('gpt-centroids')
            accs.append(acc3)

        ### classify by description score averaging
        if args.gpt_score_averaging_eval:
            scores = torch.zeros(image_features.shape[0], len(dset.classnames)).cuda()
            for i, c in enumerate(dset.classnames):
                classname = gpt_helpers.transform_classname(replace_underscores(c))
                _descriptions = gpt_descriptions[classname]
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    v = F.normalize(
                        gpt_helpers.encode_text_wrapper(
                            model_copy, _descriptions, tokenizer
                        )
                    )
                scores[:, i] = (image_features.float().cuda() @ v.T).mean(dim=1)
            acc4 = (scores.max(1).indices == y_truth.cuda()).float().mean().item() * 100.
            print('classify by description score averaging')
            metrics.append('gpt-score-averaging')
            accs.append(acc4)
        
        ### no token offset smart prompt choices 6 ensemble
        
        def _soup_evaluation(metric_name, descriptors):
            global accs, metrics, image_features, y_truth
            text_features = _get_descriptor_centroids(
                model_copy, descriptors, text, token_offsets=[0], 
                descriptor_scores=None)
            _acc = _evaluate(image_features, text_features, y_truth)
            metrics.append(metric_name)
            accs.append(_acc)
            
            if args.token_offset_eval:
                text_features = _get_descriptor_centroids(
                    model_copy, descriptors, text, 
                    token_offsets=[0, 5, 10, 15, 20, 25], 
                    descriptor_scores=None)
                _acc = _evaluate(image_features, text_features, y_truth)
                metrics.append(metric_name + '-token-offset')
                accs.append(_acc)
        
        if args.soup_eval:
            if not word_descriptors is None:
                assert not good_descriptors is None
                _soup_evaluation('word-soup', word_descriptors)
                _soup_evaluation('descriptor-soup', good_descriptors)
                # test code for:
                # get_descriptor_features(tokenized_text_with_descriptors, model, dim)
#                 init_text_features = get_descriptor_features(tokenized_text_with_descriptors, model, dim=args.d)
#                 init_text_features = F.normalize(init_text_features.mean(0))
#                 _accc = _evaluate(image_features, init_text_features, y_truth)
#                 metrics.append('word-soup')
#                 accs.append(_accc)
            else:
                _soup_evaluation('soup', descriptors)
                
        if args.score_averaging:
             ### score averaging 
            __descriptors = word_descriptors if not word_descriptors is None else descriptors
            text_features_stack = _get_descriptor_feature_matrix(
                    model_copy, __descriptors, text, 
                    token_offsets=[0]
                )
            scores = torch.zeros(image_features.shape[0], len(dset.classnames)).cuda()
            for i, c in enumerate(dset.classnames):
                v = F.normalize(text_features_stack[:, i, :])
                scores[:, i] = (image_features.float().cuda() @ v.T).mean(dim=1)
            acc7 = (scores.max(1).indices == y_truth.cuda()).float().mean().item() * 100.
            metrics.append('word-soup-score-average')
            accs.append(acc7)
            print(' +++ score averaging')
            print('acc: ', acc7)
            
        if args.zs_average:
            class_labels = get_text_labels(dset.classnames, prompt)
            for l in class_labels:
                assert l[-1] == '.'
            text_feature_stack = []
            for desc in word_descriptors:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    t_f = model.encode_text(tokenizer([si[:-1] +',' + desc for si in class_labels]).cuda())
                text_feature_stack.append(F.normalize(t_f))
#             for _ in range(len(text_feature_stack)):
#                 text_feature_stack.append(F.normalize(zs_text_features))
            text_features = F.normalize(torch.stack(text_feature_stack).mean(0))
            scores = 0.5 * (image_features.float().cuda() @ zs_text_features.float().T) + 0.5 * (image_features.float().cuda() @ text_features.float().T)
#             scores = image_features.float().cuda() @ text_features.float().T
            acc8 = (scores.max(1).indices == y_truth.cuda()).float().mean().item() * 100.
            metrics.append('ZS-average-with-word-desc')
            accs.append(acc8)

        print(accs)
        return_dict[dataset] = accs
        
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
    