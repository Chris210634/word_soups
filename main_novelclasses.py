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
parser.add_argument('--subsample_classes', default = 'base', type=str)
parser.add_argument('--openai_eval', default = 0, type=int)
parser.add_argument('--gpt_centroid_eval', default = 0, type=int)
parser.add_argument('--gpt_score_averaging_eval', default = 0, type=int)
parser.add_argument('--soup_eval', default = 0, type=int)
parser.add_argument('--token_offset_eval', default = 0, type=int)
args = parser.parse_args()
print(args)

# use default descriptor files
default_word_descriptor_file = 'cache/{}word_soup_descriptors_seed{}__{}_{}.list'.format(
    args.dataset,
    args.seed, 
    args.modelname, 
    args.pretrained
)
default_desc_descriptor_file = 'cache/{}good_descriptions_seed{}__{}_{}.list'.format(
    args.dataset,
    args.seed, 
    args.modelname, 
    args.pretrained
)

word_descriptors = torch.load(default_word_descriptor_file)
good_descriptors = torch.load(default_desc_descriptor_file)

print('CHOSEN WORD SOUP DESCRIPTORS: ')
for desc in word_descriptors:
    print(desc)

print('CHOSEN DESC SOUP ESCRIPTORS: ')
for desc in good_descriptors:
    print(desc)

############################ SETUP ############################
base_cfg = argparse.Namespace()
base_cfg.ROOT = args.data_dir
base_cfg.NUM_SHOTS = 16
base_cfg.SEED = args.seed
base_cfg.SUBSAMPLE_CLASSES = args.subsample_classes
new_cfg = deepcopy(base_cfg)
new_cfg.SUBSAMPLE_CLASSES = 'new'
device = "cuda"
bs = args.bs
modelname = args.modelname
pretrained = args.pretrained
cache_dir = args.cache_dir
d = args.d
epochs = args.n_epochs
iters_per_epoch = args.iters_per_epoch

prompt = prompt_strings[args.dataset]
dataset_class = dataset_classes[args.dataset]
####################################################################

############################ TRANSFORMS ############################
train_xform = get_train_transform()
test_xform = get_test_transform()
base_dset = dataset_class(base_cfg)
new_dset = dataset_class(new_cfg)

dset_base_train = dassl_dataset_conversion(
    base_dset, train_xform, 'train', shots=args.shots)
if not args.dataset in ['ImageNet']:
    dset_base_test = dassl_dataset_conversion(base_dset, test_xform, 'test')
    dset_new_test = dassl_dataset_conversion(new_dset, test_xform, 'test')
    new_classnames = new_dset.classnames
else:
    # unlike the other datasets, when using immagenet,
    # following standard procedure, we use the 50,000 validation images
    # for testing
    dset_base_test, dset_new_test = get_imagenet_val_dataset(
        test_xform,
        imagenet_root = os.path.join(args.data_dir, 'imagenet'))
    for i in range(500):
        assert base_dset.classnames[i] == get_imagenet_classnames()[i]
    new_classnames = get_imagenet_classnames()[500:]

if args.subsample_classes == 'base':
    # when dealing with the base2novel setup,
    # sanity check that the class sets do not overlap
    for name in base_dset.classnames:
        assert not name in new_classnames
    
print('len(dset_base_train): ', len(dset_base_train))
print('len(dset_base_test): ', len(dset_base_test))
print('len(dset_new_test): ', len(dset_new_test))
print('number of base classes: ', len(base_dset.classnames))
print('number of new classes: ', len(new_classnames))
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
dl_base_test = torch.utils.data.DataLoader(
                dset_base_test,
                num_workers=8,
                batch_size=32,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
dl_new_test = torch.utils.data.DataLoader(
                dset_new_test,
                num_workers=8,
                batch_size=32,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
####################################################################

############################ MODEL #################################
n_classes = len(base_dset.classnames)
        
### Get the zeroshot text prototypes (with prompt)
tokenizer = open_clip.get_tokenizer(modelname)
text_base = tokenizer(get_text_labels(base_dset.classnames, prompt))
text_new = tokenizer(get_text_labels(new_classnames, prompt))

model =  MyClip(modelname, pretrained, n_classes, d, 
                  temp = args.temp, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text_base,
                cache_dir=cache_dir)
model = model.cuda()

init_sd = deepcopy(model.cpu().state_dict())
model = model.cuda()
init_frozen_model = deepcopy(model)
init_frozen_model.eval()
####################################################################

############################ OPTIMIZERS ############################
if not args.eval_only:
    params = get_params(args, model)
    optimizer, scheduler, scaler = get_optimizer(args, params)
ema_model = deepcopy(model)
####################################################################

lr = args.lr
loss_list = []
accs = []
train_it = ForeverDataIterator(dl_base_train)

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
            epoch=epoch
        )
    #################################################################
    
    ema_model.eval()
    model_copy = deepcopy(ema_model)
    ft_sd = deepcopy(model_copy.cpu().state_dict())
    
    if args.save_model:
        save_path = 'cache/novelclasses_ft_sd_{}.pt'.format(round(random.random()*10000000))
        torch.save(ft_sd, save_path)
        print('saved ft_sd in {}'.format(save_path))
        
    model_copy.cuda()
    model_copy.eval()
    
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
    
    # only evaluate on novel classes if trained only on base classes
    if args.subsample_classes == 'base':
        print('Runing Evaluation ...')
        
        # Base classes
        dl = {'base':dl_base_test, 'new':dl_new_test}
        text_dic = {'base':text_base, 'new':text_new}
        accs = {'base':{}, 'new':{}}
        dsets = {'base':base_dset, 'new':new_dset}
        
        for mode in ['base','new']:
            with torch.no_grad():
                image_features, y_truth = get_features(
                    dl[mode], model_copy, d=args.d) 
            model_copy.reset_text(text_dic[mode])
                
            ### Zero-shot
            print('a photo of ZS')
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                text_features = model_copy.get_text_prototypes()
            metric_name = 'ZS'
            acc = _evaluate(image_features, text_features, y_truth)
            accs[mode][metric_name] = acc
            
            ### ensemble
            if args.openai_eval:
                text_features = gpt_helpers.get_openai_manual_prompt_template_centroids(
                    dsets[mode].classnames, model_copy, args.dataset, tokenizer
                )
                print('openai manual prompt ensemble centroid ZS')
                acc2 = _evaluate(image_features, text_features, y_truth)
                accs[mode]['ensemble'] = acc2
            
            ### centroid of gpt descriptions
            if args.gpt_centroid_eval:
                gpt_descriptions = gpt_helpers.get_gpt_descriptions(dataset=args.dataset)
                text_features = gpt_helpers.get_gpt_centroids(
                    gpt_descriptions, dsets[mode].classnames, model_copy, tokenizer)
                print('GPT descriptions centroid')
                acc3 = _evaluate(image_features, text_features, y_truth)
                accs[mode]['gpt-centroids'] = acc3
            
            ### classify by description score averaging
            if args.gpt_score_averaging_eval:
                scores = torch.zeros(image_features.shape[0], len(dsets[mode].classnames)).cuda()
                for i, c in enumerate(dsets[mode].classnames):
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
                accs[mode]['gpt-score-averaging'] = acc4
            
            ### soups
            def _soup_evaluation(metric_name, descriptors):
                global image_features, y_truth, mode, accs
                text_features = _get_descriptor_centroids(
                    model_copy, descriptors, text_dic[mode], token_offsets=[0], 
                    descriptor_scores=None)
                _acc = _evaluate(image_features, text_features, y_truth)
                accs[mode][metric_name] = _acc

                if args.token_offset_eval:
                    text_features = _get_descriptor_centroids(
                        model_copy, descriptors, text_dic[mode], 
                        token_offsets=[0, 5, 10, 15, 20, 25], 
                        descriptor_scores=None)
                    _acc = _evaluate(image_features, text_features, y_truth)
                    accs[mode][metric_name + '-token-offset'] = _acc

            if args.soup_eval:
                _soup_evaluation('word-soup', word_descriptors)
                _soup_evaluation('descriptor-soup', good_descriptors)

        for metric in accs['base'].keys():
            print(metric, end=',')
            for mode in ['base','new']:
                print(accs[mode][metric], end=',')
            print()
        
    del model_copy, ft_sd

    
    
    
    
    
    