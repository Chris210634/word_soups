import torchvision.transforms as transforms
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
import PIL
import numpy as np
import random
from copy import deepcopy

# https://github.com/dongzelian/SSF/blob/main/models/vision_transformer.py#L378
def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift

# https://github.com/dongzelian/SSF/blob/main/models/vision_transformer.py#L388
def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

# https://github.com/mlfoundations/open_clip/blob/fb72f4db1b17133befd6c67c9cf32a533b85a321/src/open_clip/transformer.py#L189
class SSFResidualAttentionBlock(
    open_clip.transformer.ResidualAttentionBlock):
    def __init__(
            self,
            resblock
    ):
        super(open_clip.transformer.ResidualAttentionBlock, self).__init__()

        self.ln_1 = resblock.ln_1
        self.attn = resblock.attn
        self.ls_1 = resblock.ls_1
        if hasattr(resblock, "ln_1_kv"):
            self.ln_1_kv = resblock.ln_1_kv

        self.ln_2 = resblock.ln_2
        self.mlp = resblock.mlp
        self.ls_2 = resblock.ls_2
        
        ln_1_out_dim = resblock.ln_1.bias.shape[0]
        attn_out_dim = resblock.attn.out_proj.bias.shape[0]
        ln_2_out_dim = resblock.ln_2.bias.shape[0]
        mlp_out_dim = resblock.mlp.c_proj.bias.shape[0]
        
        self.ssf_scale_ln_1, self.ssf_shift_ln_1 = init_ssf_scale_shift(ln_1_out_dim)
        self.ssf_scale_attn, self.ssf_shift_attn = init_ssf_scale_shift(attn_out_dim)
        self.ssf_scale_ln_2, self.ssf_shift_ln_2 = init_ssf_scale_shift(ln_2_out_dim)
        self.ssf_scale_mlp, self.ssf_shift_mlp = init_ssf_scale_shift(mlp_out_dim)

    def forward(
            self, q_x, k_x=None, v_x=None, attn_mask= None
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        
        q_x_normed = self.ln_1(q_x)
        q_x_normed = ssf_ada(q_x_normed, self.ssf_scale_ln_1, self.ssf_shift_ln_1)
        
        attn_out = self.attention(q_x=q_x_normed, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        attn_out = ssf_ada(attn_out, self.ssf_scale_attn, self.ssf_shift_attn)

        x = q_x + self.ls_1(attn_out)
        
        ln_2_out = self.ln_2(x)
        ln_2_out = ssf_ada(ln_2_out, self.ssf_scale_ln_2, self.ssf_shift_ln_2)
        mlp_out = self.mlp(ln_2_out)
        
        mlp_out = ssf_ada(mlp_out, self.ssf_scale_mlp, self.ssf_shift_mlp)
        x = x + self.ls_2(mlp_out)
        
        return x
    
class AdapterResidualAttentionBlock(
    open_clip.transformer.ResidualAttentionBlock):
    def __init__(
            self,
            resblock,
        reduction=4
    ):
        super(open_clip.transformer.ResidualAttentionBlock, self).__init__()

        self.ln_1 = resblock.ln_1
        self.attn = resblock.attn
        self.ls_1 = resblock.ls_1
        if hasattr(resblock, "ln_1_kv"):
            self.ln_1_kv = resblock.ln_1_kv

        self.ln_2 = resblock.ln_2
        self.mlp = resblock.mlp
        self.ls_2 = resblock.ls_2
        
        mlp_out_dim = resblock.mlp.c_proj.bias.shape[0]
        self.adapter = LinearAdapter(mlp_out_dim, reduction=reduction)

    def forward(
            self, q_x, k_x=None, v_x=None, attn_mask= None
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        
        q_x_normed = self.ln_1(q_x)
        attn_out = self.attention(q_x=q_x_normed, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = q_x + self.ls_1(attn_out)
        
        ln_2_out = self.ln_2(x)
        mlp_out = self.mlp(ln_2_out)
        
        ratio = 0.2
        mlp_out = ratio * self.adapter(mlp_out) + (1 - ratio) * mlp_out
        
        x = x + self.ls_2(mlp_out)
        
        return x

# https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class LinearAdapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(LinearAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class LoraMultiheadAttention(torch.nn.Module):
    def __init__(self, mhn, rank=4):
        super().__init__()
        # pytorch mhn stores the QKV projection matrices 
        # concatenated together along the first dimension
        in_dim = mhn.in_proj_weight.shape[0] // 3
        out_dim = mhn.in_proj_weight.shape[1]
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # only tune lora on Q and V matrices
        self.lora_Q_A = torch.nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_Q_B = torch.nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_V_A = torch.nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_V_B = torch.nn.Parameter(torch.zeros(rank, in_dim))
        
        nn.init.normal_(self.lora_Q_A, std=0.02)
        torch.nn.init.zeros_(self.lora_Q_B)
        nn.init.normal_(self.lora_V_A, std=0.02)
        torch.nn.init.zeros_(self.lora_V_B)
        
        self.mhn = mhn
        self.mhn_in_proj_weight = torch.clone(mhn.in_proj_weight.data).cuda()
        self.mhn_in_proj_weight.requires_grad = False
        del self.mhn.in_proj_weight
        
        self.scaling = 0.2
        
    def forward(self, q, k, v, **kwargs):
        self.mhn.in_proj_weight = self.mhn_in_proj_weight.detach()
        self.mhn.in_proj_weight[:self.in_dim, :] += (self.lora_Q_A @ self.lora_Q_B).T * self.scaling
        self.mhn.in_proj_weight[self.in_dim*2:, :] += (self.lora_V_A @ self.lora_V_B).T * self.scaling
        
        return self.mhn(q,k,v, **kwargs)

class PromptedTransformer(open_clip.transformer.Transformer):
    def __init__(
            self,
            transformer,
            d=512, # token dimension
            start_prompt_index=1, # start @ 1 because first token is <sos> or <cls>
            prompt_length=3,
            prompt_depth=3,
            maple=False
    ):
        '''Create a prompted transformer. Copy over stuff from transformer.'''
        super(open_clip.transformer.Transformer, self).__init__()
        self.width = transformer.width
        self.layers = transformer.layers
        self.grad_checkpointing = False
        self.resblocks = transformer.resblocks
        # shallow prompt has been handled already, so prompt_depth-1 prompts here
        if prompt_depth > 1 and not maple:
            self.prompts = torch.nn.Parameter(
                torch.randn(prompt_depth-1, prompt_length, d)
            )
            print('deep prompting prompt shape: ', self.prompts.data.shape)
            
            # Note regarding initialization:
            # MaPLe: nn.init.normal_(ctx_vectors, std=0.02)
            # VPT: xavier uniform init:
            # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
            #          val = sqrt(6 / (768 + 512)) = 0.068
            # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            nn.init.normal_(self.prompts, std=0.02)
            
        self.start_prompt_index = start_prompt_index
        self.d = d
        self.prompt_length = prompt_length
        self.prompt_depth = prompt_depth

    def forward(self, x: torch.Tensor, prompts=None, attn_mask=None):
        for depth, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                # add prompting before calling the residual layer
                # print(x.shape) # torch.Size([context_length, batch_size, d]) # 11 layers
                x = r(x, attn_mask=attn_mask)
            # if prompt_depth=1, then only input is prompted, so we have nothing to do here.
            if depth < self.prompt_depth-1:
                if prompts is None:
                    p = self.prompts
                else:
                    p = prompts
                prompt = p[depth, :, :].repeat(x.shape[1], 1, 1).permute(1, 0, 2)
                # prompt.shape: [prompt_length, batch_size, d]
                # replace the prompted section of x with the prompt
                x[self.start_prompt_index:self.start_prompt_index+self.prompt_length] = prompt
                
        return x

class ShallowPrompt(torch.nn.Module):
    def __init__(self, 
                 prompt_init, 
                 tokenizer,
                 tokenized_text_prototypes,
                 model, # const
                 M=None, 
                 class_specific_prompt=False,
                 rand_init=False,
                 suffix_string=''
                ):
        '''
        Prompt tuning:
        A text prototype is usually of the form:
        <sos> a photo of a <classname> <eos>
          ^   ^   ^   ^  ^     ^        ^
          |   |   |   |  |     |        |
          |   |   |   |  |    classname and end of sentence fixed.
          |   ------------- M trainable tokens.
          ----------------- start of sentence token (do not train).
        '''
        super(ShallowPrompt, self).__init__()
        self.prompt_init = prompt_init
        M = len(prompt_init.strip().split())
        self.M = M
        self.tokenizer = tokenizer
        # self.M is the length of trainable context vector
        self.reset_cls_vectors(model, tokenized_text_prototypes)
        
        print('prompt tuning with prompt length M={}'.format(M))
        ctx_vectors = model.token_embedding(
            self.tokenized_text_prototypes.to(model.text_projection.device)).detach()[0,1:M+1,:]
        print('shape of text prompt token vectors: {}'.format(ctx_vectors.shape))      
        prefix = model.token_embedding(
            self.tokenized_text_prototypes.to(model.text_projection.device)).detach()[0,0:1,:]
        print('shape of prefix token vectors: {}'.format(prefix.shape)) # sos token
        
        if rand_init:
            ctx_vectors = torch.randn_like(ctx_vectors)
        
        self.prefix = prefix
        
        self.ctx_vectors = torch.nn.Parameter(ctx_vectors)
        self.suffix_vectors = None
        
#         if len(suffix_string) > 0:
#             print('suffix string: ', suffix_string)
#             suffix_tokens = self.tokenizer([suffix_string]).view(-1)
#             print('suffic tokens: ', suffix_tokens)
#             suffix_eof = suffix_tokens.argmax(dim=-1)
#             suffix_tokens = suffix_tokens[1:suffix_eof]
#             suffix_vectors = model.token_embedding(
#                 suffix_tokens.to(model.text_projection.device)
#             ).detach()
#             print('suffix_vectors shape: ', suffix_vectors.shape)
#             self.suffix_vectors = torch.nn.Parameter(suffix_vectors)
#             self.ctx_vectors = ctx_vectors
#         else:
#             self.ctx_vectors = torch.nn.Parameter(ctx_vectors)
#             self.suffix_vectors = None

    def swap_suffix(self, suffix_string, model):
        ctx_vectors = self.ctx_vectors.data.clone()
        del self.ctx_vectors
        self.ctx_vectors = ctx_vectors
        print('suffix string: ', suffix_string)
        suffix_tokens = self.tokenizer([suffix_string]).view(-1)
        print('suffix tokens: ', suffix_tokens)
        suffix_eof = suffix_tokens.argmax(dim=-1)
        suffix_tokens = suffix_tokens[1:suffix_eof]
        suffix_vectors = model.token_embedding(
            suffix_tokens.to(model.text_projection.device)
        ).detach()
        print('suffix_vectors shape: ', suffix_vectors.shape)
        self.suffix_vectors = torch.nn.Parameter(suffix_vectors)
            
    def reset_suffix_vectors(self, suffix_vectors):
        self.suffix_vectors = suffix_vectors
        
    def reset_descriptors(self, model, descriptors):
        '''
        descriptors look like:
          which is a straight or curved shape.
        They should not start with a comma, becuase we prepend the comma in this function.
        They should end with a period
        '''
#         assert False
        assert type(descriptors) == list
        for desc in descriptors:
            assert desc[-1] == '.'
        self.descriptor_tokens = self.tokenizer(descriptors)
        self.descriptor_token_lengths = self.descriptor_tokens.argmax(dim=-1)
        
        # check that the tokenized text protos end in a period
        comma_token = 267
        period_token = 269
        eof_token = 49407
        for i in range(len(self.eof)):
#             if not self.tokenized_text_prototypes[i, self.eof[i]-1] == period_token:
#                 print('Caution! Violation of last character being a period.')
#                 print(open_clip.decode(self.tokenized_text_prototypes[i]))
            assert self.tokenized_text_prototypes[i, self.eof[i]] == eof_token
        
        desc_vectors = model.token_embedding(
            self.descriptor_tokens.to(model.text_projection.device)
        ).detach()
#         self.desc_vectors = torch.nn.Parameter(desc_vectors)
        self.desc_vectors = desc_vectors
        
        print('Calculated {} description vectors'.format(self.desc_vectors.shape[0]))
        
        # I'm going to need to chop off the period at the end of all the class texts
        # Then add a comma
        # Then add a randomly chosen description vector.
        # Be sure to modify the eof 
        
        self.comma_vector = model.token_embedding(
            torch.tensor([[comma_token]]).to(model.text_projection.device)
        ).detach()[0,0,:]
        
        self.eof_vector = model.token_embedding(
            torch.tensor([[eof_token]]).to(model.text_projection.device)
        ).detach()[0,0,:]
        
    def reset_cls_vectors(self, model, tokenized_text_prototypes):
        '''call this if classification text changes'''
        self.tokenized_text_prototypes = tokenized_text_prototypes
        self.eof = self.tokenized_text_prototypes.argmax(dim=-1)
        self.eof1 = self.tokenized_text_prototypes.argmax(dim=-1)
        self.cls_vectors = model.token_embedding(
            self.tokenized_text_prototypes.to(model.text_projection.device)
        ).detach()[:,self.M+1:,:]
        
        tokenized_prompt_init = self.tokenizer(self.prompt_init)[0][:self.M+1]
        # double check that all the text prototypes actually begin with the given prompt
        for _text in self.tokenized_text_prototypes:
            assert (tokenized_prompt_init - _text[:self.M+1]).sum() == 0
            
    def get_text_embeddings_from_tokenized_string(
        self,
        model,
        tokenized_text_prototypes
    ):
        '''
        use this function if you want to input your own prototype strings,
        (instead of using the cls_vectors saved in this class instance).
        We only want to replace the contex vectors.
        '''
        
        # just want to check that the given tokenized text prototypes
        # start with the same string this class was initialized with
        tokenized_prompt_init = self.tokenizer(self.prompt_init)[0][:self.M+1].cuda()
        for _text in tokenized_text_prototypes:
            assert (tokenized_prompt_init - _text[:self.M+1].cuda()).sum() == 0
            
        # embed the tokens and replace a photo of with learned context vecs
        emb = model.token_embedding(
            tokenized_text_prototypes.to(model.text_projection.device)
        ).detach()
        emb[:, 1:self.M+1, :] = self.ctx_vectors.repeat(
            len(tokenized_text_prototypes), 1, 1
        )
        
        eofs = tokenized_text_prototypes.argmax(dim=-1)
        return emb, eofs
    
    def insert_descriptor_embeddings(
        self,
        descriptor,
        desc_length,
        token_offset,
        emb,
        eofs
    ):
        ''' 
        insert the desecriptor embedding tokens into
        the emb (embeddings), replace period with comma.
        '''
        comma_vector = self.comma_vector
        eof_vector = self.eof_vector

        eofs = eofs + token_offset # offset descriptor
        
        period_index = eofs - 1
        assert emb.shape[0] == len(eofs)
        
        # replace period with comma
        emb[torch.arange(emb.shape[0]), period_index, :] = comma_vector.to(emb.device)
        emb[torch.arange(emb.shape[0]), eofs+desc_length, :] = eof_vector[None,:].to(emb.device)
        
        # insert descriptor into embedding
        for di in range(descriptor.shape[0]):
            emb[torch.arange(emb.shape[0]), eofs+di, :] = descriptor[di,:][None,:]
            
        # increment eof
        eofs = eofs + desc_length
        return emb, eofs 
    
    def get_text_embeddings(
        self, 
        descriptor_index=None, 
        token_embedding=None,
        token_offset=0
    ):
        eofs = self.eof
        cls_vectors = self.cls_vectors
        
        # <prefix> <ctx_vectors> <cls_vecotors>
        #                              ^
        #                              | eofs somewhere in middle
        
        emb = torch.cat((
            self.prefix.repeat(self.cls_vectors.shape[0], 
                               1, 
                               1).cuda(),
            self.ctx_vectors.repeat(self.cls_vectors.shape[0], 1, 1).cuda(),
            cls_vectors.cuda()
        ), dim=1)
        
        if descriptor_index is None and self.suffix_vectors is None:
            return emb, eofs
        
        if not self.suffix_vectors is None:
            emb, eofs = self.insert_descriptor_embeddings(
                descriptor=self.suffix_vectors.cuda(),
                desc_length=len(self.suffix_vectors),
                token_offset=token_offset,
                emb=emb,
                eofs=torch.clone(self.eof)
            )
#             torch.save((emb, eofs), 'emb.eofs.tup')
            return emb, eofs

        # desc_length includes sos token
        desc_length = self.descriptor_token_lengths[descriptor_index].item()
        
        # remove sos token
        dvec = self.desc_vectors[descriptor_index, 1:desc_length, :].to(emb.device)
        
        # desc_length includes sos token, so we subtract one here
        emb, eofs = self.insert_descriptor_embeddings(
            descriptor=dvec,
            desc_length=desc_length-1,
            token_offset=token_offset,
            emb=emb,
            eofs=torch.clone(self.eof)
        )

#         torch.save((emb, eofs), 'emb.eofs.tup') ### goes with test code in waffleCLIP.ipynb

        # TEST CODE
        # import open_clip
        # modelname='ViT-B-16'
        # tokenizer = open_clip.get_tokenizer(modelname)
        # open_clip_model, _, _ = open_clip.create_model_and_transforms(
        #     modelname, 
        #     pretrained='openai',
        #     cache_dir=cache_dir
        # )
        # token_embedding = open_clip_model.token_embedding.weight.cuda()
        # token_embedding.shape
        # def _get(ii):
        #     ctx = embs[ii]
        #     distance = torch.cdist(ctx, token_embedding)
        #     tokenized_text = distance.min(1).indices
        #     print( tokenized_text.argmax().item() , eofs[ii])
        #     print(open_clip.decode(tokenized_text))
        # import torch
        # embs, eofs = torch.load('emb.eofs.tup')
        # for i in range(10):
        #     _get(i)
    
        return emb, eofs
    
class MyClip(open_clip.model.CLIP):
    def __init__(self, 
                 modelname, 
                 pretrained,
                 n_classes, 
                 d, 
                 temp, 
                 tokenizer=None,
                 tokenized_text_prototypes=None,
                 args=None,
                 cache_dir='',
                 descriptors=None
                ):
        super(open_clip.model.CLIP, self).__init__()
        
        open_clip_model, _, _ = open_clip.create_model_and_transforms(
            modelname, 
            pretrained=pretrained,
            cache_dir=cache_dir
        )
        
        ### Original open_clip parameters
#         self.output_dict = open_clip_model.output_dict
        self.visual = open_clip_model.visual
#         self.context_length = open_clip_model.context_length
#         self.vocab_size = open_clip_model.vocab_size

        # for some reason, the open_clip implementation for text encode
        # is a little different
        # between vanilla CLIP and CoCa, so we need to handle that here
        if type(open_clip_model) == open_clip.coca_model.CoCa:
            self.token_embedding = open_clip_model.text.token_embedding
            self.positional_embedding = open_clip_model.text.positional_embedding
            self.ln_final = open_clip_model.text.ln_final
            self.text_projection = open_clip_model.text.text_projection
            self.register_buffer('attn_mask', 
                                 open_clip_model.text.attn_mask, persistent=False)
            self.cls_emb = open_clip_model.text.cls_emb
            # OpenCLIP's CoCa model has a "cls_emb" for the text transformer that
            # the vanilla OpenCLIP doesn not have
            self.pad_id = open_clip_model.text.pad_id
            self.heads = open_clip_model.text.heads
        else:
            self.token_embedding = open_clip_model.token_embedding
            self.positional_embedding = open_clip_model.positional_embedding
            self.ln_final = open_clip_model.ln_final
            self.text_projection = open_clip_model.text_projection
            self.register_buffer('attn_mask', open_clip_model.attn_mask, persistent=False)
            self.cls_emb = None
            
        self.logit_scale = open_clip_model.logit_scale
        
        ### My extra variables
        self.temp = temp
        self.W = None # used to store text prototypes during evaluation (when they remain constant). do not use during training
        
        ### CLIP-Adapter
        if args.adapter:
            self.adapter = Adapter(args.d, reduction=args.rank)
        else:
            self.adapter = None
            
        ### Regular Adapter
        if args.resblock_adapter:
            # vision backbone
            for depth, resblock in enumerate(open_clip_model.visual.transformer.resblocks):
                if depth >= args.layer_start_v:
                    print('replacing depth {} visual xformer attn with Adapter resblock'.format(depth))
                    open_clip_model.visual.transformer.resblocks[depth] = AdapterResidualAttentionBlock(
                        resblock, reduction=args.rank).cuda()
                    
            # text backbone
            if type(open_clip_model) == open_clip.coca_model.CoCa:
                for depth, resblock in enumerate(open_clip_model.text.transformer.resblocks):
                    if depth >= args.layer_start_t:
                        print('replacing depth {} text xformer attn with Adapter resblock'.format(depth))
                        open_clip_model.text.transformer.resblocks[depth] = AdapterResidualAttentionBlock(
                            resblock, reduction=args.rank).cuda()
            else:
                for depth, resblock in enumerate(open_clip_model.transformer.resblocks):
                    if depth >= args.layer_start_t:
                        print('replacing depth {} text xformer attn with Adapter resblock'.format(depth))
                        open_clip_model.transformer.resblocks[depth] = AdapterResidualAttentionBlock(
                            resblock, reduction=args.rank).cuda()
            
        ### SSF
        if args.ssf:
            # vision backbone
            for depth, resblock in enumerate(open_clip_model.visual.transformer.resblocks):
                if depth >= args.layer_start_v:
                    print('replacing depth {} visual xformer attn with SSF resblock'.format(depth))
                    open_clip_model.visual.transformer.resblocks[depth] = SSFResidualAttentionBlock(resblock).cuda()
                    
            # text backbone
            if type(open_clip_model) == open_clip.coca_model.CoCa:
                for depth, resblock in enumerate(open_clip_model.text.transformer.resblocks):
                    if depth >= args.layer_start_t:
                        print('replacing depth {} text xformer attn with SSF resblock'.format(depth))
                        open_clip_model.text.transformer.resblocks[depth] = SSFResidualAttentionBlock(resblock).cuda()
            else:
                for depth, resblock in enumerate(open_clip_model.transformer.resblocks):
                    if depth >= args.layer_start_t:
                        print('replacing depth {} text xformer attn with SSF resblock'.format(depth))
                        open_clip_model.transformer.resblocks[depth] = SSFResidualAttentionBlock(resblock).cuda() 
        
        ### LoRA
        if args.lora:
            for depth, resblock in enumerate(open_clip_model.visual.transformer.resblocks):
                print('replacing depth {} visual xformer attn with lora attn'.format(depth))
                resblock.attn = LoraMultiheadAttention(resblock.attn, rank=args.rank).cuda()
            if type(open_clip_model) == open_clip.coca_model.CoCa:
                for depth, resblock in enumerate(open_clip_model.text.transformer.resblocks):
                    print('replacing depth {} text xformer attn with lora attn'.format(depth))
                    resblock.attn = LoraMultiheadAttention(resblock.attn, rank=args.rank).cuda()
            else:
                for depth, resblock in enumerate(open_clip_model.transformer.resblocks):
                    print('replacing depth {} text xformer attn with lora attn'.format(depth))
                    resblock.attn = LoraMultiheadAttention(resblock.attn, rank=args.rank).cuda()
        
        ### Shallow prompting
        self.tokenized_text_prototypes = tokenized_text_prototypes.cuda()
        if args.text_prompt_depth > 0:
            self.shallow_prompt = ShallowPrompt(
                prompt_init=args.shallow_prompt_init,
                tokenizer=tokenizer,
                tokenized_text_prototypes=tokenized_text_prototypes,
                model=self, # const
                suffix_string=args.suffix_string
                # rand_init=bool(args.prompt_rand_init)
            )
            if not descriptors is None:
                self.shallow_prompt.reset_descriptors(self, descriptors)
        else:
            self.shallow_prompt = None
        
        ### Deep textual prompts
        self.transformer = PromptedTransformer(
            open_clip_model.text.transformer if type(open_clip_model) == open_clip.coca_model.CoCa else open_clip_model.transformer,
            d=d, # token dimension
            start_prompt_index=1, # start @ 1 because first token is <sos> or <cls>
            prompt_length=args.text_prompt_length,
            prompt_depth=args.text_prompt_depth
        )
        
        ### Deep visual prompts
        # if maple, use linear projector to calculate visual prompts from text prompts
        self.maple = bool(args.maple)
        if args.maple:
            assert args.visual_prompt_depth > 0
            self.visual_prompt_depth = args.visual_prompt_depth
            self.visual_prompt_length = args.visual_prompt_length
            assert args.text_prompt_depth == args.visual_prompt_depth
            assert args.text_prompt_length == args.visual_prompt_length
            text_token_size = d
            image_token_size = len(self.visual.class_embedding)
            self.maple_projector = torch.nn.Linear(text_token_size, image_token_size)
            if args.visual_prompt_depth > 1:
                self.visual.transformer = PromptedTransformer(
                    self.visual.transformer,
                    d=len(self.visual.class_embedding), # token dimension
                    start_prompt_index=1, # start @ 1 because first token is <cls>
                    prompt_length=self.visual_prompt_length,
                    prompt_depth=self.visual_prompt_depth,
                    maple=True
                )
        else:
            self.visual_prompt_depth = args.visual_prompt_depth
            self.visual_prompt_length = args.visual_prompt_length
            if self.visual_prompt_depth > 0:
                self.shallow_visual_prompt = torch.nn.Parameter(
                    torch.randn(self.visual_prompt_length, len(self.visual.class_embedding))
                )
                nn.init.normal_(self.shallow_visual_prompt, std=0.02)
                if self.visual_prompt_depth > 1:
                    self.visual.transformer = PromptedTransformer(
                        self.visual.transformer,
                        d=len(self.visual.class_embedding), # token dimension
                        start_prompt_index=1, # start @ 1 because first token is <cls>
                        prompt_length=self.visual_prompt_length,
                        prompt_depth=self.visual_prompt_depth
                    )
            else:
                self.shallow_visual_prompt = None
        
    def reset_text(self, tokenized_text_prototypes):
        '''call this if classification text changes'''
        self.tokenized_text_prototypes = tokenized_text_prototypes
        if not self.shallow_prompt is None:
            self.shallow_prompt.reset_cls_vectors(self, tokenized_text_prototypes)
            
    def encode_image(self, image, normalize: bool = False):
        # https://github.com/mlfoundations/open_clip/blob/24ddefb37fc4892f6a0c975b732226fe8a9a8613/src/
        # open_clip/transformer.py#L460C1-L461C1
        # using open_clip.transformer.VisionTransformer.forward
        # features = self.visual(image)
        x = image
        
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        # Removed from open_clip library as of October 2023
        if hasattr(self.visual, 'input_patchnorm') and self.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.visual.grid_size[0], 
                          self.visual.patch_size[0], self.visual.grid_size[1], self.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        # print(x.shape): torch.Size([batch_size, 196, 768])
        # print(self.visual.class_embedding.shape): torch.Size([768])
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        # print(x.shape): torch.Size([batch_size, 197, 768])

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)        

        x = x.permute(1, 0, 2)  # NLD -> LND
        # insert the prompt between the cls token and the image tokens
        if self.maple:
            projected_visual_prompt = self.maple_projector(self.shallow_prompt.ctx_vectors)
            x = torch.cat([
                x[0:1],
                projected_visual_prompt.repeat(x.shape[1], 1, 1).permute(1, 0, 2),
                x[1:]
            ])
        elif not self.shallow_visual_prompt is None:
            x = torch.cat([
                x[0:1],
                self.shallow_visual_prompt.repeat(x.shape[1], 1, 1).permute(1, 0, 2),
                x[1:]
            ])
            
        if self.maple:
            x = self.visual.transformer(x, prompts=self.maple_projector(self.transformer.prompts))
        else:
            x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        
        features = pooled
    
        return F.normalize(features, dim=-1) if normalize else features
    
    def build_cls_mask(self, cls_mask, cast_dtype: torch.dtype):
        '''
        for CoCa text encoder cls token stuff.
                https://github.com/mlfoundations/open_clip/blob/fb72f4db1b17133befd6c67c9cf32a533b85a321/src/open_clip/transformer.py#L587
        
        '''
        cls_mask = cls_mask.unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask
        
    def encode_text(self, text, 
                    normalize: bool = False,
                    embedding=False, # True if text is already embedded (as in prompt tuning)
                    eof=None # must set to index of eof character if embedding=True
                   ):
        cast_dtype = self.transformer.get_cast_dtype()

        if not embedding:
            eof = text.argmax(dim=-1)
            x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        else:
            x = text
            eof = eof.to(text.device)
            assert not eof is None
            
        def _repeat(t, N: int):
            return t.reshape(1, 1, -1).repeat(N, 1, 1)
        if self.cls_emb is not None:
            # CoCa only
            x = x[:, :-1, :] # make space for CLS token
            seq_len = x.shape[1]
            seq_len += 1
            x = torch.cat([x, _repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = (
                torch.arange(
                    text[:, :-1].shape[1]
                ) * torch.ones(text[:, :-1].shape[0], 1).int()
            ).to(eof.device) <= eof.unsqueeze(0).T
            cls_mask = self.build_cls_mask(cls_mask, cast_dtype)
            attn_mask = self.attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
            x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        else:
            x = x + self.positional_embedding.to(cast_dtype)
            attn_mask = self.attn_mask
            
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.cls_emb is not None:
            # when there is a classification token, 
            # take the features from the cls token (CoCa has it at the end)
            x = x[:, -1]
            x = self.ln_final(x)
        else:
#             x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
#             xs = []
#             for ci in range(1):
#                 # take features from the eot embedding 
#                 # (eot_token is the highest number in each sequence)
#                 xi = x[torch.arange(x.shape[0]), eof+ci]
#                 xs.append( F.normalize(xi @ self.text_projection) )
#             x = F.normalize(torch.stack(xs).mean(0))
# #         x = x @ self.text_projection

            x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding 
            # (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), eof]
            
        x = x @ self.text_projection
        
        return F.normalize(x, dim=-1) if normalize else x
        
    def get_text_prototypes(
        self, 
        autocast=True, 
        text_prototype_indices=None,
        descriptor_index=-1,
        token_offset=0
    ):
        if self.shallow_prompt is None:
            with torch.cuda.amp.autocast(enabled=bool(autocast)):
                if text_prototype_indices is not None:
                    text_prototypes = self.encode_text(
                        self.tokenized_text_prototypes.cuda()[text_prototype_indices])
                else:
                    text_prototypes = self.encode_text(self.tokenized_text_prototypes.cuda())
        else:
#             n_desc = self.shallow_prompt.desc_vectors.shape[0]
            n_classes = self.tokenized_text_prototypes.shape[0]
            if descriptor_index == -1:
                descriptor_index = None
                # descriptor_index = torch.randint(0,n_desc,size=(n_classes,))
            else:
                descriptor_index = descriptor_index
#                 descriptor_index = torch.ones((n_classes,)).long() * descriptor_index
            emb, eofs = self.shallow_prompt.get_text_embeddings(
                descriptor_index=descriptor_index, 
                token_offset=token_offset,
                token_embedding=self.token_embedding.weight.detach())
    
            with torch.cuda.amp.autocast(enabled=bool(autocast)):
                if text_prototype_indices is not None:
                    selected_emb = emb[text_prototype_indices]
                    text_prototypes = self.encode_text(
                        selected_emb.to(emb.dtype), 
                        embedding=True, 
                        eof=eofs.cuda()[text_prototype_indices]
                    )
                else:
                    text_prototypes = self.encode_text(emb, embedding=True, eof=eofs)
        return F.normalize(text_prototypes.float())
            
    def forward(self, x, return_features=False, 
                eval_text_features=False,
                return_text_prototypes=False,
                autocast=True,
                text_prototype_indices=None
               ):
        '''
        if eval_text_features==True, re-evaluate text prototypes every batch
        if return_text_prototypes==True, return the text prototypes along with predicitons.
        if text_prototype_indices is not None, expect a tensor of indices.
            This indicates the indices of the text prototypes we want. 
            (Don't evaluate the other prototypes.)
            We need this when there are a lot of classes and there is not enough memory
            to train all text protos in one mini-batch.
            
            if mixup_index is not None, then expect mixup_lam is also not None:
                In this case, the text prototypes should also be mixed up.
        '''
            
        with torch.cuda.amp.autocast(enabled=bool(autocast)):
            f = self.encode_image(x)
        f = f.float()
        
        if not self.adapter is None:
            ratio = 0.2
            f = ratio * self.adapter(f) + (1 - ratio) * f

        if eval_text_features:
            text_prototypes = self.get_text_prototypes(
                autocast=bool(autocast), 
                text_prototype_indices=text_prototype_indices,
            )
        else:
            text_prototypes = self.W
        
        if text_prototypes is None:
            y_hat = None
        else:
            y_hat = self.temp * (F.normalize(f.float()) @ F.normalize(text_prototypes).T)
        
        
        if return_features:
            if return_text_prototypes:
                return f, y_hat, F.normalize(text_prototypes) #, hidden_image, hidden_text
            return f, y_hat
        else:
            return y_hat
