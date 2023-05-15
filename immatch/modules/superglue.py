import torch
from argparse import Namespace
import numpy as np

from third_party.superglue.models.superglue import SuperGlue as SG
from third_party.superglue.models.utils import read_image
from .superpoint import SuperPoint
from .base import Matching

class SuperGlue(Matching):
    def __init__(self, args):
        super().__init__()        
        self.imsize = args['imsize']
        self.no_match_upscale = args.get('no_match_upscale', False)

        self.model = SG(args).eval().to(self.device)
        self.detector = SuperPoint(args)
        rad = self.detector.model.config['nms_radius']
        self.name = f'SuperGlue_r{rad}'
        print(f'Initialize {self.name}')
        
    def match_inputs_(self, gray1, gray2):
        # Detect SuperPoint features
        pred1 = self.detector.model({'image': gray1})
        pred2 = self.detector.model({'image': gray2})
        
        # Construct SuperGlue input
        data = {'image0': gray1, 'image1': gray2}
        pred = {k + '0': v for k, v in pred1.items()}
        pred = {**pred, **{k + '1': v for k, v in pred2.items()}}
        data = {**data, **pred}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
                
        # SuperGlue matching
        pred = {**pred, **self.model(data)}
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts1, kpts2 = pred['keypoints0'], pred['keypoints1']
        matches, scores = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        p1s = kpts1[valid]
        p2s = kpts2[matches[valid]]    
        matches = np.concatenate([p1s, p2s], axis=1)
        scores = scores[valid]
        return matches, kpts1, kpts2, scores
    
    def match_pairs(self, im1_path, im2_path):
        _, gray1, sc1 = read_image(im1_path, self.device, [self.imsize], 0, True)
        _, gray2, sc2 = read_image(im2_path, self.device, [self.imsize], 0, True)
        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2        
        return matches, kpts1, kpts2, scores
    