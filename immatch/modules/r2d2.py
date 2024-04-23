from argparse import Namespace
import torch
import numpy as np
import sys
from pathlib import Path

r2d2_path = Path(__file__).parent / "../../third_party/r2d2"
sys.path.append(str(r2d2_path))

from third_party.r2d2.extract import NonMaxSuppression, extract_multiscale
from third_party.r2d2.tools.dataloader import norm_RGB
from third_party.r2d2.nets.patchnet import *
from PIL import Image
from .base import FeatureDetection, Matching


class R2D2(FeatureDetection, Matching):
    def __init__(self, args=None):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.args = args

        # Initialize model
        ckpt = torch.load(args.ckpt)
        self.model = eval(ckpt["net"]).to(self.device).eval()
        self.model.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        )
        self.name = "R2D2"
        print(f"Initialize {self.name}")

        # Init NMS Detector
        self.detector = NonMaxSuppression(
            rel_thr=args.reliability_thr, rep_thr=args.repeatability_thr
        )

    def load_and_extract(self, im_path):
        # No image resize here
        im = Image.open(im_path).convert("RGB")
        im = norm_RGB(im)[None].to(self.device)
        kpts, desc = self.extract_features(im)
        return kpts, desc

    def extract_features(self, im):
        args = self.args
        max_size = 9999 if args.imsize < 0 else args.imsize
        xys, desc, scores = extract_multiscale(
            self.model,
            im,
            self.detector,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            min_size=args.min_size,
            max_size=max_size,
            verbose=False,
        )
        idxs = scores.argsort()[-args.top_k or None :]
        kpts = xys[idxs]
        desc = desc[idxs]
        return kpts, desc

    def match_pairs(self, im1_path, im2_path):
        kpts1, desc1 = self.load_and_extract(im1_path)
        kpts2, desc2 = self.load_and_extract(im2_path)

        # NN Match
        match_ids, scores = self.mutual_nn_match(
            desc1, desc2, threshold=self.args.match_threshold
        )
        p1s = kpts1[match_ids[:, 0], :2].cpu().numpy()
        p2s = kpts2[match_ids[:, 1], :2].cpu().numpy()
        matches = np.concatenate([p1s, p2s], axis=1)
        return matches, kpts1, kpts2, scores
