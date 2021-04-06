from abc import ABCMeta, abstractmethod
import torch
from .nn_matching import *

class FeatureDetection(metaclass=ABCMeta):
    '''An abstract class for local feature detection and description methods'''
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
                
    @abstractmethod            
    def extract_features(self, im, **kwargs):
        """Given the processed input, the keypoints and descriptors are extracted by the model.
        Return:
            kpts : a Nx2 tensor, N is the number of keypoints.
            desc : a NxD tensor, N is the number of descriptors 
                   and D is dimension of each descriptor.            
        """
        
    @abstractmethod        
    def load_and_extract(self, im_path, **kwargs):
        """Given an image path, the input image is firstly loaded and processed accordingly,  
        the keypoints and descriptors are then extracted by the model.
        Return:
            kpts : a Nx2 tensor, N is the number of keypoints.
            desc : a NxD tensor, N is the number of descriptors 
                   and D is dimension of each descriptor.            
        """
                
    def describe(self, im, kpts, **kwargs):
        """Given the processed input and the pre-detected keypoint locations,
        feature descriptors are described by the model.
        Return:
            desc : a NxD tensor, N is the number of descriptors 
                   and D is dimension of each descriptor.
        """

    def detect(self, im, **kwargs):
        """Given the processed input, the keypoints are detected by that method.
        Return:
            kpts : a Nx2 tensor, N is the number of keypoints.
        """        
        
class Matching(metaclass=ABCMeta):
    '''An abstract class for a method that perform matching from the input pairs'''
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
    
    @classmethod
    def mutual_nn_match(self, desc1, desc2, threshold=0.0):
        """The feature descriptors from the pair of images are matched 
        using nearset neighbor search with mutual check and an optional 
        outlier filtering. This is normally used by feature detection methods.
        Args:
            desc1, desc2: descriptors from the 1st and 2nd image of a pair.
            threshold: the cosine similarity threshold for the outlier filtering.            
        Return:
            match_ids: the indices of the matched descriptors.
        """
        if type(desc1) == torch.Tensor:
            match_ids = mutual_nn_matching_torch(desc1, desc2, threshold)
        else:
            match_ids = mutual_nn_matching(desc1, desc2, threshold)
        return match_ids
            
    @abstractmethod
    def match_pairs(self, im1_path, im2_path, **kwargs):
        """The model detects correspondences from a pair of images.
        All steps that are required to estimate the correspondences by a method
        are implemented here.
        Input:
            im1_path, im2_path: the paths of the input image pair.
            other args depend on the model.
            
        Return:
            matches: the detected matches stored as numpy array with shape Nx4,
                     N is the number of matches.
            kpts1, kpts2: the keypoints used for matching. For methods that don't 
                    explicitly define keypoints, e.g., SparseNCNet, 
                    the keypoints are the locations of points that get matched.
            scores: the matching score or confidence of each correspondence.
                    Notices, matching scores are defined differently across methods.
                    For NN matcher, they can be the cosine distance of descriptors;
                    For SuperGlue, they are the probablities in the OT matrix; ..etc.                    
        """
    
    
    
    
    