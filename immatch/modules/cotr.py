from argparse import Namespace
import numpy as np
import imageio
import torch
from pathlib import Path
import sys
cotr_path =  Path(__file__).parent / '../../third_party/cotr'
sys.path.append(str(cotr_path))

from COTR.models import build_model
from .base import Matching
from COTR.inference.sparse_engine import SparseEngine, FasterSparseEngine


class COTR(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.batch_size = args.batch_size
        self.max_corrs = args.max_corrs
        args.dim_feedforward = args.backbone_layer_dims[args.layer]
    
        self.model = build_model(args)
        self.model.load_state_dict(
            torch.load(args.ckpt, map_location='cpu')['model_state_dict']
        )
        self.model = self.model.eval().to(self.device)
        self.name = 'COTR'
        print(f'Initialize {self.name}')
    
    def match_pairs(self, im1_path, im2_path, queries_im1=None):
        im1 = imageio.imread(im1_path, pilmode='RGB')
        im2 = imageio.imread(im2_path, pilmode='RGB')
        engine = SparseEngine(self.model, self.batch_size, mode='tile')
        matches = engine.cotr_corr_multiscale(
            im1, im2, np.linspace(0.5, 0.0625, 4), 1,
            max_corrs=self.max_corrs, queries_a=queries_im1,
            force=True
        )

        # Fake scores as not output by the model
        scores = np.ones(len(matches))
        kpts1 = matches[:, :2]
        kpts2 = matches[:, 2:4]
        return matches, kpts1, kpts2, scores


