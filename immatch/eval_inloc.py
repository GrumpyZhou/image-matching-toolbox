import argparse
from argparse import Namespace
import os
import logging
from third_party.hloc.hloc.localize_inloc import localize_with_matcher
import yaml
import immatch

def eval_inloc(config_name, prefix=None):
    # Initialize Model
    config_file = f'configs/{config_name}.yml'
    with open(config_file, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)['inloc']
        class_name = args['class']
        print(f'Method:{class_name} Args: {args}')    
    model = immatch.__dict__[class_name](args)
    print(model)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)

    # Data setup
    dataset_dir = 'data/datasets/InLoc'
    retrieval_pairs = os.path.join('third_party/hloc/pairs/inloc', args['pairs'])

    # Output dir
    skip_matches = args['skip_matches']
    imsize = args['imsize']
    rthres = args['rthres']
    mthres = args['match_threshold']

    pair_tag = retrieval_pairs.split('query-')[-1].replace('.txt', '')
    exp_name = f'{pair_tag}_sk{skip_matches}im{imsize}rth{rthres}mth{mthres}'
    if prefix:
        exp_name = f'{exp_name}.{prefix}'
    odir = os.path.join('outputs/inloc', model.name, exp_name)
    method_tag = f'{model.name}_{exp_name}'        
    print('>>>res:', method_tag)
    # Localize InLoc queries
    localize_with_matcher(matcher, dataset_dir, retrieval_pairs, 
                          odir, method_tag,
                          rthres=rthres, skip_matches=skip_matches)    
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Localize Inloc')
    parser.add_argument('--gpu', '-gpu', type=str, default=0)
    parser.add_argument('--config', type=str, default=None)    
    parser.add_argument('--prefix', type=str, default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    config_name = args.config
    eval_inloc(config_name, args.prefix)
    