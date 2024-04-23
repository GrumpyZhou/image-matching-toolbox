from argparse import Namespace
import torch
import numpy as np
import cv2

from third_party.caps.CAPS.network import CAPSNet
from immatch.utils.data_io import load_im_tensor
from .base import FeatureDetection, Matching
from .superpoint import SuperPoint
from .sift import SIFT


class CAPS(FeatureDetection, Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        self.match_threshold = (
            args.match_threshold if "match_threshold" in args else 0.0
        )
        self.model = CAPSNet(args, self.device).eval()
        self.load_model(args.ckpt)

        # Initialize detector
        if args.detector.lower() == "sift":
            self.detector = SIFT(args)
            self.name = f"CAPS_SIFT"
        else:
            self.detector = SuperPoint(vars(args))
            rad = self.detector.model.config["nms_radius"]
            self.name = f"CAPS_SuperPoint_r{rad}"
        print(f"Initialize {self.name}")

    def load_im(self, im_path):
        return load_im_tensor(
            im_path,
            self.device,
            imsize=self.imsize,
            with_gray=True,
            raw_gray=("SIFT" in self.name),
        )

    def load_model(self, ckpt):
        print("Reloading from {}".format(ckpt))
        ckpt_dict = torch.load(ckpt)
        self.model.load_state_dict(ckpt_dict["state_dict"])

    def extract_features(self, im, gray):
        kpts = self.detector.detect(gray)
        if isinstance(kpts, np.ndarray):
            kpts = torch.from_numpy(kpts).float().to(self.device)
        desc = self.describe(im, kpts)
        return kpts, desc

    def describe(self, im, kpts):
        kpts = kpts.unsqueeze(0)
        feat_c, feat_f = self.model.extract_features(im, kpts)
        desc = torch.cat((feat_c, feat_f), -1).squeeze(0)
        return desc

    def load_and_extract(self, im_path):
        im, gray, scale = self.load_im(im_path)
        kpts, desc = self.extract_features(im, gray)
        kpts = kpts * torch.tensor(scale).to(kpts)  # N, 2
        return kpts, desc

    def match_inputs_(self, im1, gray1, im2, gray2):
        kpts1, desc1 = self.extract_features(im1, gray1)
        kpts2, desc2 = self.extract_features(im2, gray2)
        kpts1 = kpts1.cpu().data.numpy()
        kpts2 = kpts2.cpu().data.numpy()

        # NN Match
        match_ids, scores = self.mutual_nn_match(
            desc1, desc2, threshold=self.match_threshold
        )
        p1s = kpts1[match_ids[:, 0], :2]
        p2s = kpts2[match_ids[:, 1], :2]
        matches = np.concatenate([p1s, p2s], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path):
        im1, gray1, sc1 = self.load_im(im1_path)
        im2, gray2, sc2 = self.load_im(im2_path)
        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(im1, gray1, im2, gray2)
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
