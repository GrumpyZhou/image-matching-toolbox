import torch
import numpy as np

from third_party.superglue.models.superpoint import SuperPoint as SP
from .base import FeatureDetection, Matching
from immatch.utils.data_io import load_gray_scale_tensor_cv


class SuperPoint(FeatureDetection, Matching):
    def __init__(self, args=None):
        super().__init__()
        self.imsize = args["imsize"]
        self.match_threshold = (
            args["match_threshold"] if "match_threshold" in args else 0.0
        )
        self.model = SP(args).eval().to(self.device)
        rad = self.model.config["nms_radius"]
        self.name = f"SuperPoint_r{rad}"
        print(f"Initialize {self.name}")

    def load_im(self, im_path):
        return load_gray_scale_tensor_cv(im_path, self.device, imsize=self.imsize)

    def load_and_extract(self, im_path):
        gray, scale, _ = self.load_im(im_path)
        kpts, desc = self.extract_features(gray)
        kpts = kpts * torch.tensor(scale).to(kpts)  # N, 2
        return kpts, desc

    def extract_features(self, gray):
        # SuperPoint outputs: {keypoints, scores, descriptors}
        pred = self.model({"image": gray})
        kpts = pred["keypoints"][0]
        desc = pred["descriptors"][0].permute(1, 0)  # N, D
        return kpts, desc

    def detect(self, gray):
        kpts, _ = self.extract_features(gray)
        return kpts

    def match_inputs_(self, gray1, gray2):
        kpts1, desc1 = self.extract_features(gray1)
        kpts2, desc2 = self.extract_features(gray2)
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
        gray1, sc1, _ = self.load_im(im1_path)
        gray2, sc2, _ = self.load_im(im2_path)
        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
