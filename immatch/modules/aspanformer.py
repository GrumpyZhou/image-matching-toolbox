from argparse import Namespace
import torch
import numpy as np
import cv2

from third_party.aspanformer.src.ASpanFormer.aspanformer import ASpanFormer as ASpanFormer_
from third_party.aspanformer.src.config.default import get_cfg_defaults
from third_party.aspanformer.src.utils.misc import lower_config

from .base import Matching
from immatch.utils.data_io import load_gray_scale_tensor_cv

class ASpanFormer(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)

        self.imsize = args.imsize        
        self.match_threshold = args.match_threshold
        self.no_match_upscale = args.no_match_upscale

        # Load model
        conf = lower_config(get_cfg_defaults().ASPAN)
        conf['coarse']['train_res'] = args.train_res
        conf['match_coarse']['thr'] = args.match_threshold
        self.model = ASpanFormer_(config=conf)
        ckpt_dict = torch.load(args.ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'], strict=False)
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = f'ASpanFormer_{self.ckpt_name}'        
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')

    def load_im(self, im_path):
        return load_gray_scale_tensor_cv(
            im_path, self.device, imsize=self.imsize, value_to_scale=max,
        )

    def match_inputs_(self, gray1, gray2):
        batch = {'image0': gray1, 'image1': gray2}
        self.model(batch, online_resize=True)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path):
        gray1, sc1 = self.load_im(im1_path)
        gray2, sc2 = self.load_im(im2_path)
        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)
        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
