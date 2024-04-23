from argparse import Namespace
import torch
from pathlib import Path
import sys
import numpy as np

sparsencnet_path = Path(__file__).parent / "../../third_party/sparsencnet"
sys.path.append(str(sparsencnet_path))

from third_party.sparsencnet.lib.model import ImMatchNet
from third_party.sparsencnet.lib.normalization import imreadth, resize, normalize
from third_party.sparsencnet.lib.sparse import get_matches_both_dirs, unique
from third_party.sparsencnet.lib.relocalize import (
    relocalize,
    relocalize_soft,
    eval_model_reloc,
)
from .base import Matching


def load_im(im_path, scale_factor, imsize=None):
    im = imreadth(im_path)
    h, w = im.shape[-2:]
    if not imsize:
        imsize = max(h, w)
    else:
        imsize = imsize
    im = resize(normalize(im), imsize, scale_factor)
    return im, h, w


class SparseNCNet(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.args = args
        self.match_threshold = args.match_threshold
        self.model = self.init_model(args)
        self.name = f"SparseNCNet"
        if args.Npts:
            self.name += f"_N{args.Npts}"
        print(f"Initialize {self.name}")

    def init_model(self, args):
        chp_args = torch.load(args.ckpt)["args"]
        model = ImMatchNet(
            use_cuda=torch.cuda.is_available(),
            checkpoint=args.ckpt,
            ncons_kernel_sizes=chp_args.ncons_kernel_sizes,
            ncons_channels=chp_args.ncons_channels,
            sparse=True,
            symmetric_mode=bool(chp_args.symmetric_mode),
            feature_extraction_cnn=chp_args.feature_extraction_cnn,
            bn=bool(chp_args.bn),
            k=chp_args.k,
            return_fs=True,
            change_stride=args.change_stride,
        )

        scale_factor = 0.0625
        if args.relocalize == 1:
            scale_factor = scale_factor / 2
        if args.change_stride == 1:
            scale_factor = scale_factor * 2
        args.scale_factor = scale_factor
        return model

    def match_pairs(self, im1_path, im2_path):
        args = self.args

        im1, hA, wA = load_im(im1_path, args.scale_factor, args.imsize)
        im2, hB, wB = load_im(im2_path, args.scale_factor, args.imsize)
        # print('Ims', im1.shape, im2.shape)
        corr4d, feature_A_2x, feature_B_2x, fs1, fs2, fs3, fs4 = eval_model_reloc(
            self.model, {"source_image": im1, "target_image": im2}, args
        )
        xA_, yA_, xB_, yB_, score_ = get_matches_both_dirs(corr4d, fs1, fs2, fs3, fs4)

        if args.Npts is not None:
            matches_idx_sorted = torch.argsort(-score_.view(-1))
            N_matches = min(args.Npts, matches_idx_sorted.shape[0])
            matches_idx_sorted = matches_idx_sorted[:N_matches]
            score_ = score_[:, matches_idx_sorted]
            xA_ = xA_[:, matches_idx_sorted]
            yA_ = yA_[:, matches_idx_sorted]
            xB_ = xB_[:, matches_idx_sorted]
            yB_ = yB_[:, matches_idx_sorted]

        if args.relocalize:
            fs1, fs2, fs3, fs4 = 2 * fs1, 2 * fs2, 2 * fs3, 2 * fs4
            # relocalization stage 1:
            if args.reloc_type.startswith("hard"):
                xA_, yA_, xB_, yB_, score_ = relocalize(
                    xA_,
                    yA_,
                    xB_,
                    yB_,
                    score_,
                    feature_A_2x,
                    feature_B_2x,
                    crop_size=args.reloc_hard_crop_size,
                )
                if args.reloc_hard_crop_size == 3:
                    _, uidx = unique(
                        yA_.double() * fs2 * fs3 * fs4
                        + xA_.double() * fs3 * fs4
                        + yB_.double() * fs4
                        + xB_.double(),
                        return_index=True,
                    )
                    xA_ = xA_[:, uidx]
                    yA_ = yA_[:, uidx]
                    xB_ = xB_[:, uidx]
                    yB_ = yB_[:, uidx]
                    score_ = score_[:, uidx]
            elif args.reloc_type == "soft":
                xA_, yA_, xB_, yB_, score_ = relocalize_soft(
                    xA_, yA_, xB_, yB_, score_, feature_A_2x, feature_B_2x
                )

            # relocalization stage 2:
            if args.reloc_type == "hard_soft":
                xA_, yA_, xB_, yB_, score_ = relocalize_soft(
                    xA_,
                    yA_,
                    xB_,
                    yB_,
                    score_,
                    feature_A_2x,
                    feature_B_2x,
                    upsample_positions=False,
                )

            elif args.reloc_type == "hard_hard":
                xA_, yA_, xB_, yB_, score_ = relocalize(
                    xA_,
                    yA_,
                    xB_,
                    yB_,
                    score_,
                    feature_A_2x,
                    feature_B_2x,
                    upsample_positions=False,
                )

        yA_ = (yA_ + 0.5) / (fs1)
        xA_ = (xA_ + 0.5) / (fs2)
        yB_ = (yB_ + 0.5) / (fs3)
        xB_ = (xB_ + 0.5) / (fs4)

        xA = xA_.view(-1).data.cpu().float().numpy() * wA
        yA = yA_.view(-1).data.cpu().float().numpy() * hA
        xB = xB_.view(-1).data.cpu().float().numpy() * wB
        yB = yB_.view(-1).data.cpu().float().numpy() * hB
        scores = score_.view(-1).data.cpu().float().numpy()

        matches = np.stack((xA, yA, xB, yB), axis=1)
        kpts1 = matches[:, :2]
        kpts2 = matches[:, 2:]
        return matches, kpts1, kpts2, scores
