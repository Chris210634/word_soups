import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default = '', type =str) # dataset to train on
    parser.add_argument('--bs', default = 64, type =int)       # batch size
    parser.add_argument('--seed', default = 1, type =int)     # random seed (1, 2, or 3)
    parser.add_argument('--accum_iter', default = 1, type =int) # for simulated large batch
    parser.add_argument('--lr', default = 2e-5, type =float)    # learning rate
    parser.add_argument('--wd', default = 1e-5, type =float)    # weight decay (l2 reg)
    parser.add_argument('--shots', default = 16, type =int)
    
    parser.add_argument('--d', default = 512, type =int) # dimension of CLIP embedding
    parser.add_argument('--modelname', default = 'ViT-B-16', type =str)
    parser.add_argument('--pretrained', default = 'openai', type =str)
    parser.add_argument('--cache_dir', default = "", type =str)
    parser.add_argument('--data_dir', default = "", type =str)
    
    parser.add_argument('--n_epochs', default = 1, type = int)
    parser.add_argument('--iters_per_epoch', default = 750, type = int) # different for every dataset
    parser.add_argument('--lr_decay', default = 0.0, type =float) # learning rate decay. 0 means no lr decay.
    parser.add_argument('--temp', default = 60., type =float)     # inverse of softmax temperature

    parser.add_argument('--optimizer', default = 'sgd', type = str) # sgd or adam    
    parser.add_argument('--loss', default = 'ce', type = str)       # name of loss function
    parser.add_argument('--label_smoothing', default = 0.0, type=float)

    parser.add_argument('--samples_per_class', default = 1, type=int)
    parser.add_argument('--margin', default = 0.0, type=float)
    parser.add_argument('--adaptive_margin', default = 0.0, type=float)
    parser.add_argument('--ema', default = 0.995, type=float)
    
    parser.add_argument('--visual_prompt_depth', default = 0, type=int)
    parser.add_argument('--visual_prompt_length', default = 3, type=int)
    parser.add_argument('--text_prompt_length', default = 3, type=int)
    parser.add_argument('--text_prompt_depth', default = 1, type=int)
    parser.add_argument('--train_text_encoder', default = 1, type=int)
    parser.add_argument('--train_visual_encoder', default = 1, type=int)
    parser.add_argument('--layer_start_v', default = 9, type = int)   # only optimize later xformer layers
    parser.add_argument('--layer_start_t', default = 9, type = int)
    parser.add_argument('--maple', default = 0, type = int)
    parser.add_argument('--lora', default = 0, type = int)
    parser.add_argument('--bitfit', default = 0, type = int)
    parser.add_argument('--adapter', default = 0, type = int)
    parser.add_argument('--ssf', default = 0, type = int)
    parser.add_argument('--resblock_adapter', default = 0, type = int)
    
    parser.add_argument('--rank', default = 4, type = int) # use for lora, adapter, and CLIP_adapter
    
    parser.add_argument('--shallow_prompt_init', default = "a photo of", type=str)
    parser.add_argument('--prompt_lr_multi', default = 10.0, type=float)
    parser.add_argument('--eval_only', default = 0, type=int)
    parser.add_argument('--save_model', default = 0, type=int)
    
    parser.add_argument('--n_descriptors', default = -1, type=int)
    
    parser.add_argument('--checkpoint', default = '', type=str)
    parser.add_argument('--prompt_rand_init', default = 0, type=int)
    parser.add_argument('--suffix_string', default = '', type=str)
    
    parser.add_argument('--train_with_descriptors', default = 0, type=int)
    parser.add_argument('--teacher_temp', default = 100.0, type=float)
    parser.add_argument('--init_lam', default = 0.0, type=float)
    parser.add_argument('--skip_ema_iters', default = 0, type=int)
    
    return parser
