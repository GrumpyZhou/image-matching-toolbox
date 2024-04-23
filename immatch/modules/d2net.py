from argparse import Namespace
import torch
import numpy as np
import sys
from pathlib import Path

d2net_path = Path(__file__).parent / "../../third_party/d2net"
sys.path.append(str(d2net_path))

from third_party.d2net.lib.model_test import D2Net as D2N
from third_party.d2net.lib.utils import preprocess_image
from third_party.d2net.lib.pyramid import process_multiscale
from immatch.utils.data_io import read_im
from .base import FeatureDetection, Matching


class D2Net(FeatureDetection, Matching):
    def __init__(self, args=None):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.model = D2N(
            model_file=args.ckpt,
            use_relu=args.use_relu,
            use_cuda=torch.cuda.is_available(),
        )
        self.ms = args.multiscale
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.name = "D2Net"
        print(f"Initialize {self.name}")

    def load_and_extract(self, im_path):
        im, scale = read_im(im_path, self.imsize)
        im = np.array(im)
        im = preprocess_image(im, preprocessing="caffe")
        kpts, desc = self.extract_features(im)
        kpts = kpts * scale  # N, 2
        return kpts, desc

    def extract_features(self, im):
        if self.ms:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    im[np.newaxis, :, :, :].astype(np.float32), device=self.device
                ),
                self.model,
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    im[np.newaxis, :, :, :].astype(np.float32), device=self.device
                ),
                self.model,
                scales=[1],
            )

        kpts = keypoints[:, [1, 0]]  # (x, y) and remove the scale
        desc = descriptors
        return kpts, desc

    def match_pairs(self, im1_path, im2_path):
        kpts1, desc1 = self.load_and_extract(im1_path)
        kpts2, desc2 = self.load_and_extract(im2_path)

        # NN Match
        match_ids, scores = self.mutual_nn_match(
            desc1, desc2, threshold=self.match_threshold
        )
        p1s = kpts1[match_ids[:, 0], :2]
        p2s = kpts2[match_ids[:, 1], :2]
        matches = np.concatenate([p1s, p2s], axis=1)
        return matches, kpts1, kpts2, scores
