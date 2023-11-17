import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
import PIL
import numpy as np
import random
from copy import deepcopy

class EMA:
    '''
    This class implements the Expontial Moving Average (EMA) for model weights.
    Only used for evaluation.
    Using the EMA averaged model increases the quality of generated images.
    '''
    def __init__(self, beta=0.995):
        '''
        beta is a hyperparameter.
        New model weights = beta * (old model weights) + 
                            (1 - beta) * (new model weights)
        '''
        super().__init__()
        self.beta = beta

    def step_ema(self, ma_model, current_model):
        '''
        ma_model: the averaged model we will use for evaluation
        current_model: The model being explicitly trained
        This function updates the weights of ma_model. Return None.
        '''
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        '''Private function used to update individual parameters.'''
        return old * self.beta + (1 - self.beta) * new

# def multimodal_loss(f_visual, f_text, 
#                     logit_scale=100., force_diag=None,
#                     balanced=False, margin=0.0):
#     '''
#     Recommended usage:
#     f_visual = F.normalize(f_visual)
#     f_text = F.normalize(text_encoder.prototypes[y.cuda()])
#     loss += multimodal_loss(f_visual, f_text, logit_scale=temp)
    
#     *** EVERYTHING MUST BE L2 NORMALIZED ***
#     '''
#     labels = torch.arange(f_text.shape[0]).cuda()
#     assert f_visual.shape[0] == f_text.shape[0]
    
#     logits_per_image = logit_scale * f_visual @ f_text.T
    
# #     if not force_diag is None:
# #         for i in range(f_text.shape[0]):
# #             logits_per_image[i,i] = logit_scale * force_diag[i]
    
#     if balanced:
#         logits_per_text = logit_scale * f_text @ f_visual.T
#         contrastive_loss = (
#                 F.cross_entropy(logits_per_image, labels) +
#                 F.cross_entropy(logits_per_text, labels)
#             ) / 2.
#         return contrastive_loss
    
#     return F.cross_entropy(logits_per_image, labels)

def multimodal_loss(f_visual, f_text, 
                    logit_scale=100.,
                    balance=0.5, 
                    margin=0.0, 
                    adaptive_margin=0.0,
                    f_visual_cache=None,
                    f_text_cache=None):
    '''
    Recommended usage:
    f_visual = F.normalize(f_visual)
    f_text = F.normalize(text_encoder.prototypes[y.cuda()])
    loss += multimodal_loss(f_visual, f_text, logit_scale=temp)
    
    *** EVERYTHING MUST BE L2 NORMALIZED ***
    '''
    assert f_visual.shape[0] == f_text.shape[0]
    
    if not f_visual_cache is None:
        assert f_visual_cache.shape == f_text_cache.shape
        _label = F.one_hot(
            torch.arange(f_visual.shape[0]), num_classes=f_visual_cache.shape[0]+f_visual.shape[0]
        ).to(f_visual.device)
        logits_per_image = logit_scale * f_visual @ torch.cat((f_text, f_text_cache)).T
        logits_per_text = logit_scale * f_text @ torch.cat((f_visual, f_visual_cache)).T
    else:
        _label = torch.eye(f_text.shape[0], device=f_visual.device)
        logits_per_image = logit_scale * f_visual @ f_text.T
        logits_per_text = logit_scale * f_text @ f_visual.T
    
    if adaptive_margin > 0.0:
        assert f_visual_cache is None
        logits_per_image = logits_per_image + logit_scale * adaptive_margin * (1.0 - f_text @ f_text.T).detach()
        
    contrastive_loss = (
            balance * my_cross_entropy(logits_per_image, _label, margin=logit_scale * margin) +
            (1.0 - balance) * my_cross_entropy(logits_per_text, _label, margin=logit_scale * margin)
        )
    return contrastive_loss

def multimodal_loss_with_cache(
    f_visual, f_text, 
    f_visual_cache, f_text_cache, 
    logit_scale=100.,
    balance=0.5
):
    '''
    *** EVERYTHING MUST BE L2 NORMALIZED ***
    '''
    _label = F.one_hot(torch.arange(f_visual.shape[0]), num_classes=f_visual_cache.shape[0]).to(f_visual.device)
    assert f_visual.shape[0] == f_text.shape[0]
    
    logits_per_image = logit_scale * f_visual @ f_text_cache.T
#     logits_per_text = logit_scale * f_text @ f_visual_cache.T # everything here detached
    assert balance == 1.0
    contrastive_loss = balance * my_cross_entropy(logits_per_image, _label, margin=0.0) 
    return contrastive_loss

def get_reg_loss(p1, p2, n, reg_type='l2'):
    norms = []
    if reg_type == 'l2':
        normp = 2
    else:
        assert reg_type == 'l1'
        normp = 1
    for i in range(len(p1)):
        norms.append((p1[i] - p2[i]).norm(p=normp))
    return torch.stack(norms).norm(p=normp).pow(normp) #/ float(n)

def get_onehot(y, n_classes, label_smoothing=0.0):
    y_onehot = torch.zeros(len(y), n_classes).to(y.device) + (label_smoothing / (n_classes-1))
    y_onehot = y_onehot.scatter_(1, y.unsqueeze(1), 1. - label_smoothing)
    return y_onehot
    
def my_cross_entropy(x, y, margin=0.0):
    '''y is one-hot. x is temperature-scaled cosine similarity.'''
    loss = - F.log_softmax(x - y * margin, dim=-1) * y
#     loss = - F.log_softmax(x, dim=-1) * y
    loss = loss.sum(-1)
    return loss.mean()

def alignment_loss(x, y):
    return -(x * y).sum(1).mean()

def multimodal_uniformity(x, y):
#     assert x.shape == y.shape
    S = torch.exp(- x @ y.T)
    S_mean = S.sum() / float(S.shape[0]) / float(S.shape[1])
#     S_mean = (S.sum() - S.diag().sum()) / float(S.shape[0]) / float(S.shape[0] - 1)
    return S_mean.log()

def uniformity_loss(x):
    S = torch.exp(- x @ x.T)
#     for i in range(S.shape[0]):
#         S[i,i] = 0.0
    S_mean = (S.sum() - S.diag().sum()) / float(S.shape[0]) / float(S.shape[0] - 1)
    return S_mean.log()

def margin_contrastive(features, prototypes, w, temp, pos_margins=1.0, neg_margins=0.6):
    scores = features @ prototypes.T
    L_pos = F.relu(pos_margins - scores) * w
    L_neg = F.relu(scores - neg_margins) * (1. - w)
    
    if (L_pos > 0.).sum() < 1e-5:
        l_p = torch.tensor(0.)
    else:
        l_p = L_pos.sum() / (L_pos > 0.).sum()
    if (L_neg > 0.).sum() < 1e-5:
        l_n = torch.tensor(0.)
    else:
        l_n = L_neg.sum() / (L_neg > 0.).sum()
        
    return l_p + l_n

def margin_hinge(features, prototypes, w, temp, margin=0.1):
    scores = features @ prototypes.T
    loss = (margin - (scores * w).sum(1, keepdim=True) + scores).square() * (1. - w)
    return loss.mean()
    
def proda_loss(image_features, text_features, logit_scale=100., margin=0.0, lam=0.1):
    '''
    text_features: F.normalized text features
    shape: n_class x n_prompt x dim
    If using the clip loss to save memory, then the first dim can be batch size.
    
    image_features must be normalized.
    '''
    # text_features = text_features.view(n_class, n_prompt, -1)
    text_mean = text_features.mean(dim=1)
    logits = logit_scale * image_features @ F.normalize(text_mean).T
    batch_size = image_features.shape[0]
    n_prompt = text_features.shape[1]
                
    text_features = text_features - text_mean.unsqueeze(1)
    diag_cov_martix = text_features.permute(2,0,1) @ text_features.permute(2,1,0)
    diag_cov_martix /= n_prompt + 1
    refined_logits = torch.einsum("bd, dik -> bik", [image_features**2, diag_cov_martix])

    # originally, the second part of sigme has torch.arange(n_class)
    # since we're using the clip loss, we replace this with torch.arange(batch_size)
    
    # original:
#     sigma = refined_logits[torch.arange(batch_size), labels, labels].unsqueeze(-1) + \
#         refined_logits[:, torch.arange(n_class), torch.arange(n_class) ] - \
#         2 * refined_logits[torch.arange(batch_size), labels, : ]

    assert text_features.shape[0] == batch_size
    # adaptation to clip loss:
    sigma = refined_logits[torch.arange(batch_size), torch.arange(batch_size), torch.arange(batch_size)].unsqueeze(-1) + \
        refined_logits[:, torch.arange(batch_size), torch.arange(batch_size) ] - \
        2 * refined_logits[torch.arange(batch_size), torch.arange(batch_size), : ]

    logits += 0.5*(logit_scale**2)*sigma.view(-1, batch_size)
    
    _label = torch.eye(batch_size, device=image_features.device)
    contrastive_loss = my_cross_entropy(logits, _label, margin=logit_scale * margin)

    nc_text_features = F.normalize(text_features.mean(dim=0))
    assert nc_text_features.shape[0] == n_prompt
    dis = nc_text_features @ nc_text_features.T
    loss_m = dis[~torch.eye(n_prompt, dtype=torch.bool, device='cuda')].abs().mean()
    
    return contrastive_loss + lam * loss_m
    
    