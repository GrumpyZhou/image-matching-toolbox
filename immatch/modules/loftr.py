from argparse import Namespace
import torch
import numpy as np
import cv2
from third_party.loftr.src.loftr import LoFTR as LoFTR_, default_cfg
from .base import *
from ..utils.data_io import read_im_gray_divisible

class LoFTR(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)

        self.imsize = args.imsize        
        self.match_threshold = args.match_threshold
        
        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        self.model = LoFTR_(config=conf)
        ckpt_dict = torch.load(args.ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'])
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = f'LoFTR_{self.ckpt_name}'        
        print(f'Initialize {self.name}')
        
    def match_pairs(self, im1_path, im2_path):
        # Load images
        im1, sc1 = read_im_gray_divisible(im1_path, self.device, 
                                          imsize=self.imsize, dfactor=8)
        im2, sc2 = read_im_gray_divisible(im2_path, self.device, 
                                          imsize=self.imsize, dfactor=8)
        
        # Matching
        batch = {'image0': im1, 'image1': im2}
        self.model(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1 * sc1, kpts2 * sc2], axis=1)        
        return matches, kpts1, kpts2, scores