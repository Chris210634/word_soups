import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
import PIL
import numpy as np
import random
import os
from copy import deepcopy

def replace_underscores(s):
        new_s = ''
        for si in s:
            if si == '_':
                new_s += ' '
            else:
                new_s += si
        return new_s

def get_captions_from_file(fn):
    captions = []
    with open(fn) as f:
        for line in f:
            if line.strip() == '':
                continue
            assert line.strip()[-5:] == '<eol>'
            captions.append(line.strip()[:-5].strip())
    return captions

def dassl_dataset_conversion(dset, xform, mode, shots=None):
    '''
    Notes:
    
    type(dset) can be anything, e.g. datasets.imagenet.ImageNet
    dset.train_x and dset.test are lists containing objexts of type
    dassl.data.datasets.base_dataset.Datum
    each Datum has attributes label and impath indicating the label 
    and absolute path the image file, respectively.
    
    These splits are pregenerated and saved in the folder 
    ``datasets/splits/<dataset name>/*.pkl``
    '''
    modes = {'test':dset.test, 'val':dset.val, 'train':dset.train_x}
    labels = [t.label for t in modes[mode]]
    impaths = [t.impath for t in modes[mode]]
    imgs = [(a, b) for a,b in zip(impaths, labels)]
    
    if shots is not None:
        print('CAUTION ! limiting dataset to {} shots. This should only be done on informal experiments.'.format(shots))
        labels_dic = {}
        for impath, label in imgs:
            if not label in labels_dic:
                labels_dic[label] = [impath]
            else:
                labels_dic[label].append(impath)
        imgs = []
        for label in labels_dic:
            assert len(labels_dic[label]) >= shots
            random.shuffle(labels_dic[label])
            impaths_sub = labels_dic[label][:shots]
            for impath in impaths_sub:
                imgs.append((impath, label))
                
    return BaseDataset(imgs, xform)

def get_imagenet21k(
    xform,
    shots=16, 
    root,
    wordnet_tree_dic_filename='imagenet21k_miil_tree.pth',
    imagenet1k_filename='classnames.txt'
):
    '''
    return a tuple of (dset, classnames).
    where dset is a PyTorch dataset with shots per class.
    where classnames is a list of classnames corresponding to each class.
    root is the folder containing classes. 
    There are K folders under "root", where K is number of classes.
    Name of each fodler looks like "n12008487"
    
    wordnet_tree_dic_filename contains the pickled dictionary containing
    the wordnet hierarchy corresponding to each class.
    (see paper <<ImageNet21k pretraining for the masses>>)
    
    Filter out imagenet1k classes so we're not seeing those.
    '''
    wordnet_tree_dic = torch.load('imagenet21k_miil_tree.pth')
    classids = os.listdir(root)
    classids.sort()
    
    # Filter out imagenet1k classes so we're not seeing those.
    filtered_classids = []
    imagenet1k_classids = []
    with open(imagenet1k_filename) as f:
        for line in f:
            imagenet1k_classid = line.strip().split(' ')[0].strip()
            if 'n' in imagenet1k_classid:
                imagenet1k_classids.append(imagenet1k_classid)
    assert len(imagenet1k_classids) == 1000
    for class_id in classids:
        if not class_id in imagenet1k_classids:
            filtered_classids.append(class_id)
    classids = filtered_classids
    classids.sort()
    
    imgs = []
    for i, classid in enumerate(classids):
        label = i
        classpath = root + '/' + classid + '/'
        jpegs = os.listdir(classpath)
        random.shuffle(jpegs)
        jpegs = jpegs[:shots]
        for jpeg in jpegs:
            impath = classpath + jpeg
            imgs.append((impath, label))
            
    class_description = wordnet_tree_dic['class_description']
    classnames = []
    for class_id in classids:
        assert classid in class_description
        classnames.append(replace_underscores(class_description[class_id]))
            
    return BaseDataset(imgs, xform), classnames

class FastRandomSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dset, batch_size, num_iters, samples_per_class=0):
        self.length = len(dset)
        self.num_batches = len(dset) // batch_size
        self.batch_size = batch_size
        self.num_times = (num_iters // self.num_batches) + 1
        self.samples_per_class = samples_per_class
        self.labels = [tup[1] for tup in dset.imgs]
        self.label_dic = {}
        for index, label in enumerate(self.labels):
            if label in self.label_dic:
                self.label_dic[label].append(index)
            else:
                self.label_dic[label] = [index]

    def __len__(self):
        return len(self.ret)

    def __iter__(self):
        self.ret = []
        for _ in range(self.num_times):
            if self.samples_per_class == 0:
                self.ret.extend(
                    list(
                        torch.randperm(self.length)[:self.num_batches * self.batch_size]
                    )
                )
            else:
                # sample self.batch_size // self.samples_per_class random classes
                # then sample self.samples_per_class from each class
                label_dic_copy = deepcopy(self.label_dic)
                for label in label_dic_copy:
                    random.shuffle(label_dic_copy[label])
                label_list = deepcopy(list(label_dic_copy.keys()))
                random.shuffle(label_list)
                ptr = 0
                for _ in range(self.num_batches * self.batch_size // self.samples_per_class):
                    chosen_label = label_list[ptr]
                    for _ in range(self.samples_per_class):
                        self.ret.append(label_dic_copy[chosen_label].pop())
                    ptr += 1
                    if ptr == len(label_list):
                        ptr = 0
                        random.shuffle(label_list)
                
        return iter(self.ret)
    
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transform, metadata=None):
        self.transform = transform
        self.imgs = imgs
        self.metadata = metadata

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        def img_load(index):
            imraw = PIL.Image.open(self.imgs[index][0])
            # convert gray to rgb
            if len(list(imraw.split())) == 1 : imraw = imraw.convert('RGB')
            if imraw.mode != 'RGB' : imraw = imraw.convert('RGB')
            if self.transform is not None:
                im = self.transform(imraw)
            return im

        im = img_load(index)
        target = self.imgs[index][1]
        
        if self.metadata is None:
            return im, target
        else:
            return im, target, self.metadata[index]
        
def concat_datasets(dset1, dset2):
    '''concatenate the two datasets of type BaseDataset. '''
    imgs = deepcopy(dset1.imgs)
    imgs.extend(deepcopy(dset2.imgs))
    assert dset1.transform == dset2.transform
    xform = dset1.transform
    assert dset1.metadata is None
    assert dset2.metadata is None
    return BaseDataset(imgs, xform, None)
        
class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 caption_file, 
                 ys, 
                 tokenizer, 
                 already_tokenized=False,
                imgs=None,
                transform=None,
                text=None):
        if text is None:
            if not already_tokenized:
                captions = get_captions_from_file(caption_file)
                self.text = tokenizer(captions)
            else:
                self.text = torch.load(caption_file)
        else:
            self.text = text
        self.ys = ys
        self.imgs = imgs
        self.transform = transform
        if not self.imgs is None:
            assert not self.transform is None
            assert len(self.text) == len(self.imgs)
#             assert len(self.ys) == len(self.imgs)
#             for i in range(len(self.ys)):
#                 assert self.ys[i] == self.imgs[i][1]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        def img_load(index):
            imraw = PIL.Image.open(self.imgs[index][0])
            if self.transform is not None:
                im = self.transform(imraw)
            return im
        
        if not self.imgs is None:
            im = img_load(index)
            target = self.imgs[index][1]
            return im, self.text[index,:], target
        
        return self.text[index,:], self.ys[index]
    
def build_confident_dataset(datasets, test_domain, confidences, threshold=0.5):
    '''
    return a dataset of type BaseDataset, that contains all the domains
    in 'datasets' other than test_domain index
    as long as the confidence is above the threshold.
    confidences is a list of same length as datasets.
    confidences is a list of pytorch tensors.
    '''
    num_domains = len(datasets)
    assert num_domains == len(confidences)
    
    dataset_imgs = []
    for d in range(num_domains):
        if d == test_domain:
            continue
        assert len(confidences[d]) == len(datasets[d])
        confident_indices = (confidences[d] > threshold).nonzero().view(-1)
        for ind in confident_indices:
            dataset_imgs.append(datasets[d].imgs[ind])
        print('Domain {} selected {} images {}%'.format(
            d, 
            len(confident_indices), 
            len(confident_indices)/len(datasets[d]) * 100.
        ))
        
    return BaseDataset(dataset_imgs, transform=datasets[0].transform)

def get_confidences(pretrained_linear, n_domains, use_goundtruth=False, temp=100.):
    with torch.no_grad():
        text_features = torch.load(pretrained_linear).data
        confidences = []
        f_all, y_all = torch.load('domainnet_clip_vit_b_16.fy')
        for d in range(n_domains):
            f = F.normalize(f_all[d])
            y_hat = temp * (f.cuda() @ text_features.T.cuda())
            y_hat = y_hat.softmax(dim=-1)
            if use_goundtruth:
                confidences.append(y_hat.gather(1, y_all[d].cuda().unsqueeze(0).T).view(-1).cpu())
            else:
                confidences.append(y_hat.max(1).values.cpu())
    return confidences

def build_confidence_thresholded_dataset(dataset, confidences, low=0.0, high=1.0):
    '''
    return a dataset of type BaseDataset.
    '''   
    dataset_imgs = []

    assert len(confidences) == len(dataset.imgs)
    selected_indices = ((confidences > low).float() * (confidences < high).float()).nonzero().view(-1)
    for ind in selected_indices:
        dataset_imgs.append(dataset.imgs[ind])
    print('selected {} images {}%'.format(
        len(selected_indices), 
        len(selected_indices)/len(dataset.imgs) * 100.
    ))
        
    return BaseDataset(dataset_imgs, transform=dataset.transform)