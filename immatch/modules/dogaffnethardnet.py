from argparse import Namespace
import torch
import numpy as np
import cv2
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from .base import FeatureDetection, Matching
from ..utils.data_io import read_im_gray


class DogAffNetHardNet(FeatureDetection, Matching):
    def __init__(self, args=None):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        try:
            self.device = args.device
        except:
            self.device = torch.device("cpu")
        self.match_threshold = args.match_threshold
        self.det = cv2.SIFT_create(
            args.npts, contrastThreshold=-10000, edgeThreshold=-10000
        )
        self.desc = KF.HardNet(True).eval().to(self.device)
        self.aff = KF.LAFAffNetShapeEstimator(True).eval().to(self.device)
        self.name = f"DoG{args.npts}-AffNet-HardNet"
        print(f"Initialize {self.name}")

    def load_im(self, im_path):
        im, scale = read_im_gray(im_path, self.imsize)
        im = np.array(im)
        return im, scale

    def load_and_extract(self, im_path):
        im, scale = self.load_im(im_path)
        kpts, desc = self.extract_features(im)
        kpts = kpts * scale
        return kpts, desc

    def extract_features(self, im):
        kpts = self.det.detect(im, None)
        # We will not train anything, so let's save time and memory by no_grad()
        with torch.no_grad():
            timg = K.image_to_tensor(im, False).float() / 255.0
            timg = timg.to(self.device)
            if timg.shape[1] == 3:
                timg_gray = K.rgb_to_grayscale(timg)
            else:
                timg_gray = timg
            # kornia expects keypoints in the local affine frame format.
            # Luckily, kornia_moons has a conversion function
            lafs = laf_from_opencv_SIFT_kpts(kpts, device=self.device)
            lafs_new = self.aff(lafs, timg_gray)
            patches = KF.extract_patches_from_pyramid(timg_gray, lafs_new, 32)
            B, N, CH, H, W = patches.size()
            # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
            # So we need to reshape a bit :)
            descs = (
                self.desc(patches.view(B * N, CH, H, W))
                .view(B * N, -1)
                .detach()
                .cpu()
                .numpy()
            )
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        return kpts, descs

    def load_and_detect(self, im_path):
        im, scale = self.load_im(im_path)
        kpts, desc = self.detect(im)
        kpts = kpts * scale
        return kpts

    def detect(self, im):
        kpts = self.det.detect(im)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        return kpts

    def match_inputs_(self, im1, im2):
        kpts1, desc1 = self.extract_features(im1)
        kpts2, desc2 = self.extract_features(im2)

        # NN Match
        dists, match_ids = KF.match_smnn(
            torch.from_numpy(desc1), torch.from_numpy(desc2), self.match_threshold
        )
        match_ids = match_ids.data.numpy()
        p1s = kpts1[match_ids[:, 0], :2]
        p2s = kpts2[match_ids[:, 1], :2]
        matches = np.concatenate([p1s, p2s], axis=1)
        scores = 1.0 - dists
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path):
        im1, sc1 = self.load_im(im1_path)
        im2, sc2 = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(im1, im2)
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
