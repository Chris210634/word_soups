import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
import PIL
import os
import numpy as np
import random
from copy import deepcopy
import math
from source.samplers import BaseDataset

def get_harmonic_mean(x, y):
    '''Harmonic mean.'''
    return 2.0 * x * y / (x+y)

def replace_underscores(s):
        new_s = ''
        for si in s:
            if si == '_':
                new_s += ' '
            else:
                new_s += si
        return new_s
    
def get_text_labels(classnames, prompt_init = 'a photo of {}'):
    text_labels = []
    for l in classnames:
        text_labels.append(prompt_init.format(replace_underscores(l.strip())))
    return text_labels

def get_zero_shot_accs(model, tokenizer, f, y, prompt_string, classnames):
    text = tokenizer(get_text_labels(classnames, prompt_string))
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = F.normalize(
            model.encode_text(text.cuda()).float().cpu()
        )
    assert text_features.shape[0] == len(classnames)

    test_preds = (F.normalize(f).cuda() @ text_features.T.cuda()).max(1).indices.cpu()
    acc = (test_preds == y).float().mean().item() * 100.
    return acc
        
class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)
    
def get_imagenet_classnames():
    '''actual names'''
    imagenet_classnames = []
    imagenet_classnames_file = 'source/classnames.txt'
    with open(imagenet_classnames_file) as f:
        for line in f:
            imagenet_classnames.append(' '.join(line.strip().split(' ')[1:]))
    return imagenet_classnames
    
def get_imagenet_labelstrings():
    '''strings like n01440764'''
    imagenet_classnames = []
    imagenet_classnames_file = 'source/classnames.txt'
    with open(imagenet_classnames_file) as f:
        for line in f:
            imagenet_classnames.append(line.split(' ')[0])
    return imagenet_classnames
    
def get_imagenet_val_dataset(
    xform, 
    imagenet_root,
    imagenet_val_csv = 'source/LOC_val_solution.csv',
    split=True
):
    imagenet_val_root = os.path.join(imagenet_root, 'images/val')
    paths = []
    ys = []
    imagenet_classnames = get_imagenet_labelstrings()
    with open(imagenet_val_csv) as f:
        for line in f:
            if not 'ImageId,PredictionString' in line:
                paths.append(os.path.join(imagenet_val_root , line.split(',')[0] + '.JPEG'))
                label_string = line.split(',')[1].split(' ')[0]
                ys.append(imagenet_classnames.index(label_string))
    imgs = [(a, b) for a,b in zip(paths, ys)]
    if not split:
        return BaseDataset(imgs, xform)
    imgs_base = []
    imgs_new = []
    for (a,b) in imgs:
        if b < 500:
            imgs_base.append((a,b))
        else:
            imgs_new.append((a,b-500))
    return BaseDataset(imgs_base, xform), BaseDataset(imgs_new, xform)
        
def tokenize_list(captions, tokenizer, n_shards=64):
    '''captions: list of string to tokenize.'''
    inc = len(captions) // n_shards
    text = []
    for shard in tqdm(range(n_shards+1)):
        begin = shard*inc
        end = (shard + 1)*inc if shard < n_shards else len(captions)
        text.append(tokenizer(captions[begin:end]))
    text = torch.cat(text)
    return text

def get_save_path(fn):
    with open(fn) as f:
        for line in f:
            if 'Save path:' in line:
                return line.strip().split(' ')[-1].strip()

def get_open_clip_features(dl, model, d=512):
    n = len(dl.dataset)
    features = torch.zeros(n, d)
    labels = torch.zeros(n).long()

    p = 0
    for x, y_true in tqdm(dl):
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                z = model.encode_image(x.cuda())
                z = z.cpu()
        b = len(y_true)
        assert z.shape[0] == b
        features[p : p+b, :] = z
        labels[p : p+b] = y_true
        p += b
    assert p == n
    return features, labels

def get_clip_text_features(text, model, bs=128, d=512):
    '''
    Be sure to tokenize prior to using this function:
    ```text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)```
    '''
    n = len(text)
    features = torch.zeros(n, d)
    n_batches = (n // bs) + 1
    p = 0
    for i in tqdm(range(n_batches)):
        begin = i*bs
        end = (i+1)*bs if i < n_batches-1 else n
        x = text[begin:end]
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                z = model.encode_text(x.cuda())
                z = z.cpu()
        b = z.shape[0]
        assert z.shape[0] == b
        features[p : p+b, :] = z
        p += b
    assert p == n
    return features

def get_features(dl, model, d=512):
    n = len(dl.dataset)
    features = torch.zeros(n, d)
    labels = torch.zeros(n).long()

    p = 0
    for x, y_true in tqdm(dl):
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                z = model(x.cuda(), return_features=True)
                if type(z) is tuple:
                    z = z[0]
                z = z.cpu()
        b = len(y_true)
        assert z.shape[0] == b
        features[p : p+b, :] = z
        labels[p : p+b] = y_true
        p += b
    assert p == n
    return features, labels

def get_open_clip_predictions(dl, model, W):
    n = len(dl.dataset)
    y_hat = torch.zeros(n).long()
    y_truth = torch.zeros(n).long()
    
    p = 0
    with torch.no_grad():
        for x, y in tqdm(dl):
            with torch.cuda.amp.autocast(enabled=True):
                f = F.normalize(model.encode_image(x.cuda()))
            b = len(y)
            assert f.shape[0] == b
            y_hat[p : p+b] = (f.float() @ W.T).max(1).indices.cpu()
            y_truth[p : p+b] = y.cpu()
            p += b
        
    assert p == n
    return y_hat, y_truth

def get_predictions(dl, model):
    n = len(dl.dataset)
    y_hat = torch.zeros(n).long()
    y_truth = torch.zeros(n).long()
    
    p = 0
    with torch.no_grad():
        for x, y in tqdm(dl):
            with torch.cuda.amp.autocast(enabled=True):
                z = model(x.cuda())
            b = len(y)
            assert z.shape[0] == b
            y_hat[p : p+b] = z.max(1).indices.cpu()
            y_truth[p : p+b] = y.cpu()
            p += b
        
    assert p == n
    return y_hat, y_truth

def interpolate_state_dicts(sd1, sd2, alpha=0.5):
    '''For Wise-FT.'''
    keys = list(sd1.keys())
    for key in keys:
        assert key in sd2
    assert len(sd1) == len(sd2)
    sd3 = deepcopy(sd1)
    for key in keys:
        sd3[key] = alpha * sd1[key] + (1. - alpha) * sd2[key]
    return sd3

def encode_text(x, model, eof):
    '''For prompt tuning.'''
    with torch.cuda.amp.autocast(enabled=True):
        # eof = text.argmax(dim=-1)
        cast_dtype = model.transformer.get_cast_dtype()
        x = x + model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x, attn_mask=model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eof] @ model.text_projection
        return x

### DomainNet classnames
# classnames = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']