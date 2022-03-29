from argparse import Namespace
import numpy as np
import imageio
import torch
from pathlib import Path
import sys
cotr_path =  Path(__file__).parent / '../../third_party/cotr'
sys.path.append(str(cotr_path))

from COTR.models import build_model
from COTR.inference.sparse_engine import SparseEngine, FasterSparseEngine


class COTR(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.ksize = args.ksize
        self.model = load_model(args.ckpt, method='cotr')
        self.name = 'COTR'
        print(f'Initialize {self.name}')
    
    def match_pairs(self, im1_path, im2_path, queries_im1=None):
        matches, scores, _ = estimate_matches(self.model, 
                                              im1_path, im2_path,
                                              ksize=self.ksize,
                                              io_thres=self.match_threshold, 
                                              eval_type='fine', 
                                              imsize=self.imsize)    
        kpts1 = matches[:, :2]
        kpts2 = matches[:, 2:4]
        return matches, kpts1, kpts2, scores


