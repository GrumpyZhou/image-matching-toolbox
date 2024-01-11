from argparse import Namespace
import torch
import numpy as np
import cv2
import torch.nn.functional as F

from third_party.aspanformer.src.ASpanFormer.aspanformer import ASpanFormer as ASpanFormer_
from third_party.aspanformer.src.config.default import get_cfg_defaults
from third_party.aspanformer.src.utils.misc import lower_config

from .base import Matching
from immatch.utils.data_io import load_gray_scale_tensor_cv
from third_party.aspanformer.src.utils.dataset import read_megadepth_gray


class ASpanFormer(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)

        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.no_match_upscale = args.no_match_upscale
        self.online_resize = args.online_resize
        self.im_padding = False
        self.coarse_scale = args.coarse_scale
        self.eval_coarse = args.eval_coarse

        # Load model
        config = get_cfg_defaults()
        conf = lower_config(config)['aspan']
        conf['coarse']['train_res'] = args.train_res
        conf['coarse']['test_res'] = args.test_res
        conf['coarse']['coarsest_level'] = args.coarsest_level
        conf['match_coarse']['border_rm'] = args.border_rm
        conf['match_coarse']['thr'] = args.match_threshold

        if args.test_res:
            self.imsize = args.test_res[::-1]
            self.im_padding = args.test_res[0] == args.test_res[1]
        self.model = ASpanFormer_(config=conf)
        ckpt_dict = torch.load(args.ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'], strict=False)
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = f'ASpanFormer_{self.ckpt_name}'        
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name} {args} ')

    def load_im(self, im_path):
        return load_gray_scale_tensor_cv(
            im_path,
            self.device,
            dfactor=8,
            imsize=self.imsize,
            value_to_scale=max,
            pad2sqr=self.im_padding
        )

    def match_inputs_(self, gray1, gray2, mask1=None, mask2=None):
        batch = {
            'image0': gray1, 'image1': gray2
        }
        if mask1 is not None and mask2 is not None and self.coarse_scale:
            [ts_mask_1, ts_mask_2] = F.interpolate(
                torch.stack([mask1, mask2], dim=0)[None].float(),
                scale_factor=self.coarse_scale,
                mode='nearest',
                recompute_scale_factor=False
            )[0].bool().to(self.device)
            batch.update({'mask0': ts_mask_1.unsqueeze(0), 'mask1': ts_mask_2.unsqueeze(0)})

        # Forward pass
        self.model(batch, online_resize=self.online_resize)

        # Output parsing
        if self.eval_coarse:
            kpts1 = batch['mkpts0_c'].cpu().numpy()
            kpts2 = batch['mkpts1_c'].cpu().numpy()
        else:
            kpts1 = batch['mkpts0_f'].cpu().numpy()
            kpts2 = batch['mkpts1_f'].cpu().numpy()
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path):
        gray1, sc1, mask1 = self.load_im(im1_path)
        gray2, sc2, mask2 = self.load_im(im2_path)
        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(
            gray1, gray2, mask1, mask2
        )

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
