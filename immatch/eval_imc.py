import argparse
from argparse import Namespace
import os
from pathlib import Path

import immatch
import immatch.utils.imc_helper as imc
from immatch.utils.model_helper import init_model

    
def eval_imc(args):
    # Initialize model
    model, model_conf = init_model(args.config, args.benchmark_name)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)
    
    # Merge the 
        
    # Init paths
    dataset_dir = Path(args.data_root) / args.dataset
    output_dir = Path('outputs') / args.benchmark_name / model.name / f'im{args.imsize}' / args.dataset
    
    # Compute matches, keypoints and descriptors
    imc.prepare_submission_custom_matchers(
        matcher, dataset_dir, output_dir, 
        args.config
        sc_thres=agrs.sc_thres, gv_thres=args.gv_thres, 
        qt_psize=args.qt_psize, qt_dthres=args.qt_dthres,
        qt_unique=args.qt_unique, skip_matching=args.skip_matching
    )
    
    # Config submission
    imc.generate_submission_json(args.config, )


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Evaluate on Image Matching Challenge')
    parser.add_argument('--gpu', '-gpu', type=str, default=0)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--skip_match', action='store_true')
    parser.add_argument('--data_root', type=str, default='data/datasets/imc-2021')
    parser.add_argument('--benchmark_name', type=str, default='imc')
    parser.add_argument(
        '--dataset', type=str, default='phototourism',
        choices=['phototourism', 'pragueparks', 'googleurban']  
    )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_imc(args)

