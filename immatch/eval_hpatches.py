import argparse
from argparse import Namespace
import os

import immatch
from immatch.utils.data_io import lprint
import immatch.utils.hpatches_helper as helper
from immatch.utils.model_helper import parse_model_config

def eval_hpatches(
    root_dir,
    config_list,
    task='both',
    h_solver='degensac',
    ransac_thres=2,
    match_thres=None,
    odir='outputs/hpatches',
    save_npy=False,
    print_out=False,
    debug=False,
):
    # Init paths
    data_root = os.path.join(root_dir, 'data/datasets/hpatches-sequences-release')
    cache_dir = os.path.join(root_dir, odir, 'cache')
    result_dir = os.path.join(root_dir, odir, 'results', task)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)    
        
    
    # Iterate over methods
    for config_name in config_list:
        # Load model
        args = parse_model_config(config_name, 'hpatch', root_dir)
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
            
            # Eval on the specified task(s)
            helper.eval_hpatches(
                matcher,
                data_root,
                model.name,
                task=task,
                h_solver=h_solver,
                ransac_thres=ransac_thres,
                lprint_=lprint_,
                print_out=print_out,
                save_npy=result_npy,
                debug=debug,
            )
        log.close()

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Benchmark HPatches')
    parser.add_argument('--gpu', '-gpu', type=str, default=0)
    parser.add_argument('--root_dir', type=str, default='.')  
    parser.add_argument('--odir', type=str, default='outputs/hpatches')
    parser.add_argument('--config', type=str,  nargs='*', default=None)    
    parser.add_argument('--match_thres', type=float, nargs='*', default=None)
    parser.add_argument(
        '--task', type=str, default='homography', 
        choices=['matching', 'homography', 'both']
    )
    parser.add_argument(
        '--h_solver', type=str, default='degensac',
        choices=['degensac', 'cv']
    )
    parser.add_argument('--ransac_thres', type=float, default=2)
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--print_out', action='store_true')
    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_hpatches(
        args.root_dir, args.config, args.task,
        h_solver=args.h_solver,
        ransac_thres=args.ransac_thres,
        match_thres=args.match_thres,
        odir=args.odir,
        save_npy=args.save_npy,
        print_out=args.print_out
    )

    

