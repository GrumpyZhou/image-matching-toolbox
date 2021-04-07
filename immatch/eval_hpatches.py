import argparse
from argparse import Namespace
import os
import yaml

import immatch
import immatch.utils.hpatches_helper as helper

def lprint(ms, log=None):
    '''Print message on console and in a log file'''
    print(ms)
    if log:
        log.write(ms+'\n')
        log.flush()
        
def eval_hpatches(root_dir, config_list, task='both', 
                  match_thres=None,save_npy=False,  print_out=False):
    
    # Init paths
    data_root = os.path.join(root_dir, 'data/datasets/hpatches-sequences-release')
    cache_dir = os.path.join(root_dir, 'outputs/hpatches/cache')
    result_dir = os.path.join(root_dir, 'outputs/hpatches/results', task)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)    
        
    
    # Iterate over methods
    for config_name in config_list:
        config_file = f'{root_dir}/configs/{config_name}.yml'
        with open(config_file, 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['hpatch']
            if 'ckpt' in args:
                args['ckpt'] = os.path.join(root_dir, args['ckpt'])
                if 'coarse' in args and 'ckpt' in args['coarse']:
                    args['coarse']['ckpt'] = os.path.join(
                        root_dir, args['coarse']['ckpt']
                    )
            class_name = args['class']
        
        # One log file per method
        log_file = os.path.join(result_dir, f'{class_name}.txt')        
        log = open(log_file, 'a')
        lprint_ = lambda ms: lprint(ms, log)

        # Iterate over matching thresholds
        thresholds = match_thres if match_thres else [args['match_threshold']] 
        lprint_(f'\n>>>> Method={class_name} Default config: {args} '
                f'Thres: {thresholds}')        
        
        for thres in thresholds:
            args['match_threshold'] = thres   # Set to target thresholds
            
            # Init model
            model = immatch.__dict__[class_name](args)
            matcher = lambda im1, im2: model.match_pairs(im1, im2)
            
            # Init result save path (for matching results)            
            result_npy = None            
            if save_npy:
                result_tag = model.name
                if args['imsize'] > 0:
                    result_tag += f".im{args['imsize']}"
                if thres > 0:
                    result_tag += f'.m{thres}'
                result_npy = os.path.join(cache_dir, f'{result_tag}.npy')
            
            lprint_(f'Matching thres: {thres}  Save to: {result_npy}')
            
            # Eval on target tasks:
            if task == 'homography':
                helper.eval_hpatches_homography(                
                    matcher, data_root, model.name, 
                    lprint_=lprint_, print_out=print_out
                ) 
            elif task == 'matching':
                helper.eval_hpatches_matching(
                    matcher, data_root, model.name, 
                    lprint_=lprint_, print_out=print_out, 
                    save_npy=result_npy, 
                )
            else:                               
                # Perform both tasks at once
                helper.eval_hpatches(
                    matcher, data_root, model.name, 
                    lprint_=lprint_, print_out=print_out, 
                    save_npy=result_npy
                )
    
        log.close()

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Benchmark HPatches')
    parser.add_argument('--gpu', '-gpu', type=str, default=0)
    parser.add_argument('--root_dir', type=str, default='.')  
    parser.add_argument('--config', type=str,  nargs='*', default=None)    
    parser.add_argument('--match_thres', type=float, nargs='*', default=None)
    parser.add_argument(
        '--task', type=str, default='homography', 
        choices=['matching', 'homography', 'both']        
    )
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--print_out', action='store_true')    
    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_hpatches(
        args.root_dir, args.config, args.task,                   
        match_thres=args.match_thres, save_npy=args.save_npy, 
        print_out=args.print_out
    )

    

