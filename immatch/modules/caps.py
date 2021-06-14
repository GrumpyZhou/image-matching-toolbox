from argparse import Namespace
import torch
import numpy as np
import cv2
from .base import *
from third_party.caps.CAPS.network import CAPSNet
from .superpoint import SuperPoint
from immatch.utils.data_io import load_im_tensor

class CAPS(FeatureDetection, Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold if 'match_threshold' in args else 0.0
        self.model = CAPSNet(args, self.device).eval()
        self.load_model(args.ckpt)
        
        # Initialize detector
        if 'detector' not in args:
            args.detector = 'SuperPoint'
        if args.detector.lower() == 'sift':
            self.detector = cv2.SIFT_create()
            self.is_sift = True            
            self.name = f'CAPS_SIFT'            
        else:
            self.detector = SuperPoint(args.__dict__)
            self.is_sift = False        
            rad = self.detector.model.config['nms_radius']
            self.name = f'CAPS_SuperPoint_r{rad}'
        print(f'Initialize {self.name}')
        
    def load_model(self, ckpt):
        print("Reloading from {}".format(ckpt))
        ckpt_dict = torch.load(ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'])        

    def extract_features(self, im, gray):
        kpts = self.detector.detect(gray)
        if self.is_sift:
            kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
            kpts = torch.from_numpy(kpts).float().to(self.device)
        desc = self.describe(im, kpts)
        return kpts, desc
        
    def describe(self, im, kpts):
        kpts = kpts.unsqueeze(0)
        feat_c, feat_f = self.model.extract_features(im, kpts)
        desc = torch.cat((feat_c, feat_f), -1).squeeze(0)
        return desc   
    
    def load_and_extract(self, im_path):
        im, gray, scale = load_im_tensor(im_path, self.device, imsize=self.imsize, 
                                         with_gray=True, raw_gray=self.is_sift)
        kpts, desc = self.extract_features(im, gray)
        kpts = kpts * torch.tensor(scale).to(kpts) # N, 2
        return kpts, desc
    
    def match_inputs_(self, im1, gray1, im2, gray2):
        kpts1, desc1 = self.extract_features(im1, gray1)
        kpts2, desc2 = self.extract_features(im2, gray2)
        kpts1 = kpts1.cpu().data.numpy()
        kpts2 = kpts2.cpu().data.numpy()
        
        # NN Match
        match_ids, scores = self.mutual_nn_match(desc1.cpu().data.numpy(), 
                                         desc2.cpu().data.numpy(), 
                                         threshold=self.match_threshold)
        p1s = kpts1[match_ids[:, 0], :2]
        p2s = kpts2[match_ids[:, 1], :2]
        matches = np.concatenate([p1s, p2s], axis=1)
        return matches, kpts1, kpts2, scores
    
    def match_pairs(self, im1_path, im2_path):
        im1, gray1, sc1 = load_im_tensor(im1_path, self.device, imsize=self.imsize, 
                                         with_gray=True, raw_gray=self.is_sift)
        im2, gray2, sc2 = load_im_tensor(im2_path, self.device, imsize=self.imsize, 
                                         with_gray=True, raw_gray=self.is_sift)
        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(im1, gray1, im2, gray2)
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
    