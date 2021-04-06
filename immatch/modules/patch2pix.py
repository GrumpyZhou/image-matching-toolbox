from argparse import Namespace
import torch
from pathlib import Path
import sys
patch2pix_path = Path(__file__).parent / '../../third_party/patch2pix'
sys.path.append(str(patch2pix_path))

from .base import *
from third_party.patch2pix.utils.eval.model_helper import load_model, estimate_matches
import immatch
from immatch.utils.data_loading import load_im_tensor


class Patch2Pix(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.ksize = args.ksize
        self.model = load_model(args.ckpt, method='patch2pix')
        self.name = 'Patch2Pix'        
        print(f'Initialize {self.name}')      
    
    def match_pairs(self, im1_path, im2_path):
        matches, scores, _ = estimate_matches(self.model, 
                                              im1_path, im2_path,
                                              ksize=self.ksize,
                                              io_thres=self.match_threshold, 
                                              eval_type='fine', 
                                              imsize=self.imsize)    
        kpts1 = matches[:, :2]
        kpts2 = matches[:, 2:4]
        return matches, kpts1, kpts2, scores

class NCNet(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.ksize = args.ksize
        self.model = load_model(args.ckpt, method='nc')
        self.name = 'NCNet'
        print(f'Initialize {self.name}')
    
    def match_pairs(self, im1_path, im2_path):
        matches, scores, _ = estimate_matches(self.model, im1_path, im2_path,                    
                                         ksize=self.ksize, ncn_thres=self.match_threshold, 
                                         eval_type='coarse', imsize=self.imsize)
        kpts1 = matches[:, :2]
        kpts2 = matches[:, 2:4]
        return matches, kpts1, kpts2, scores    
    
class Patch2PixRefined(Matching):
    def __init__(self, args):
        super().__init__()        
        # Initialize coarse matcher
        cargs = args['coarse']
        self.cname = cargs['name']
        self.cmatcher = immatch.__dict__[self.cname](cargs)
        
        # Initialize patch2pix
        args = Namespace(**args)
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.fmatcher = load_model(args.ckpt, method='patch2pix')
        self.name = f'Patch2Pix_{self.cmatcher.name}_m'+ str(cargs['match_threshold'])
        print(f'Initialize {self.name}')
    
    def match_pairs(self, im1_path, im2_path):
        im1, grey1, sc1 = load_im_tensor(im1_path, self.device, imsize=self.imsize, with_gray=True)
        im2, grey2, sc2 = load_im_tensor(im2_path, self.device, imsize=self.imsize, with_gray=True)
        if self.cname in ['SuperGlue']:
            coarse_match_res = self.cmatcher.match_inputs_(grey1, grey2)
        elif self.cname == 'CAPS':
            coarse_match_res = self.cmatcher.match_inputs_(im1, grey1, im2, grey2)
        coarse_matches = coarse_match_res[0]
        
        # Patch2Pix refinement
        refined_matches, scores, _ = self.fmatcher.refine_matches(im1, im2, 
                                                                  coarse_matches,
                                                                  io_thres=self.match_threshold)        
        upscale = np.array([sc1 + sc2])
        matches = upscale * refined_matches
        kpts1 = matches[:, :2]
        kpts2 = matches[:, 2:4]
        return matches, kpts1, kpts2, scores 



