import argparse
from argparse import Namespace
import os

from third_party.hloc.hloc.localize_inloc import localize_with_matcher
from immatch.utils.model_helper import init_model

def eval_inloc(args):
    model, model_conf = init_model(args.config, args.benchmark_name)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)
    args = Namespace(**vars(args), **model_conf)
    print(">>>>", args)

    # Setup output dir
    rthres = args.rthres
    mthres = args.match_threshold
    skip_matches = args.skip_matches
    retrieval_pairs = os.path.join(args.pair_dir, args.benchmark_name, args.pairs)
    pair_tag = retrieval_pairs.split('query-')[-1].replace('.txt', '')
    exp_name = f'{pair_tag}_sk{skip_matches}im{args.imsize}rth{rthres}mth{mthres}'
    if args.prefix:
        exp_name = f'{exp_name}.{args.prefix}'
    odir = os.path.join('outputs', args.benchmark_name, model.name, exp_name)
    method_tag = f'{model.name}_{exp_name}'        
    print('>>>Method tag:', method_tag)

    # Localize InLoc queries
    localize_with_matcher(
        matcher, args.dataset_dir, retrieval_pairs, odir,
        method_tag, rthres=rthres, skip_matches=skip_matches
    )
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Localize Inloc')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--config', type=str, default=None)    
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, default='data/datasets/InLoc')
    parser.add_argument('--pair_dir', type=str, default='data/pairs')
    parser.add_argument('--benchmark_name', type=str, default='inloc')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_inloc(args)
    