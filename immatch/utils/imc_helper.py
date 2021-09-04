from argparse import Namespace
import os
import numpy as np
from tqdm import tqdm
import h5py

from immatch.utils.localize_sfm_helper import (
    match_pairs_with_keys_exporth5,
    matches_to_keypoint_ids,
)

def generate_submission_json(config, results_dir):
    import json
    
    # TODO: 
    # 1. Parse the contents from yaml or load the json template
    # 2. Update the json according to the method & match filtering 
    # 3. Save config.json to the submission_dir (results_dir)
    pass

def filter_matches(matches, scores, sc_thres=-1, gv_thres=-1, gv_max_iters=1e5):
    """Filter raw matches.
    Args:
        matches: N, 4, input matches        
        scores: N, matching scores
        sc_thres: float, matching score threshold
        gv_thres: float, geometrical verification ransac threshold
    """
    import pydegensac
    
    # Score filtering
    inls = np.where(scores >= sc_thres)[0]
    matches = matches[inls]
    scores = scores[inls]

    # Geometric verification
    if gv_thres > 0:
        try:
            F, inls = pydegensac.findFundamentalMatrix(
                matches[:, :2], matches[:, 2:], 
                gv_thres, 0.999999, int(gv_max_iters)
            )
            matches = matches[inls]
            scores = scores[inls]
        except: 
            matches = scores = []
    return matches, scores  
    
def parse_matches_and_keypoints(raw_match_file, results_dir, sc_thres, gv_thres,
                                qt_dthres=-1, qt_psize=-1, qt_unique=True):
    '''Filter matches and extract keypoints and descriptors'''
    num_matches = []
    num_keypoints = []
    all_kp_data = {}
    with h5py.File(raw_match_file, 'r') as fraw:
        with h5py.File(os.path.join(results_dir, 'matches.h5'), 'w') as fres:
            pair_keys = list(fraw.keys())
            print(f'Start parse {len(pair_keys)} matched pairs and quantize keypoints ...')
            for key in tqdm(pair_keys, smoothing=.1):
                name0, name1 = key.split('-')

                # Load matches and scores
                matches = fraw[key]['matches'].__array__()
                scores = fraw[key]['scores'].__array__()

                # Score filtering + geometric verification 
                matches, scores = filter_matches(matches, scores, sc_thres, gv_thres)
                num_matches.append(len(matches))

                # Convert matches to keypoint ids (with optional quantize)
                match_ids = matches_to_keypoint_ids(
                    matches, scores, name0, name1, all_kp_data,                 
                    qt_dthres=qt_dthres, qt_psize=qt_psize, qt_unique=qt_unique
                )

                # Save matches            
                fres[key] = match_ids.T  # 2, N, required by imc
            print(f'Finished parsing matches for {len(fres)} pairs, matches num:{np.mean(num_matches)}')

    # Save keypoints & dummy descriptors
    with h5py.File(os.path.join(results_dir, 'keypoints.h5'), 'w') as fkps: 
        with h5py.File(os.path.join(results_dir, 'descriptors.h5'), 'w') as fdes:
            print(f'Save keypoints from {len(all_kp_data)} images...')
            for name in tqdm(all_kp_data, smoothing=.1):
                kps = np.array(all_kp_data[name]['kps'], dtype=np.float32)  # N, 2
                num_keypoints.append(len(kps))
                fkps[name] = kps  # N, 2
                
                # Dummy descriptors
                fdes[name] = np.empty([1, 2])                
            print(f'Finished quantization, kps mean/max:{np.mean(num_keypoints)}/{np.max(num_keypoints)}')
            
def prepare_submission_custom_matchers(matcher, dataset_dir, output_dir, 
                                       sc_thres=-1, gv_thres=0.5, qt_psize=-1, 
                                       qt_dthres=-1, qt_unique=True, 
                                       skip_matching=False):
    """Prepare submission for custom matchers.
    For methods such as superglue, correspondence networks, 
    *+patch2pix, Loftr..
    """

    # Experiment tag
    matches_tag = f'gv{gv_rthres}sc{sc_thres}'
    qt_tag = ''
    if qt_size > 0:
        qt_tag += f'qt{qt_psize}d{qt_dthres}'
        if qt_unique:
            qt_tag += 'uni'
    exp_tag = matches_tag + qt_tag

    # Initialize results dir
    results_dir = output_dir / exp_tag
    results_dir.mkdir(parents=True, exist_ok=True)

    # Start prepare submission per scene
    scenes = os.listdir(dataset_dir)
    for scene in scenes:
        # Set image dir
        scene_dir = os.path.join(dataset_dir, scene)
        if 'set_100' in os.listdir(scene_dir):
            scene_dir = os.path.join(scene_dir, 'set_100')
            im_dir = os.path.join(scene_dir, 'images')
        elif 'set_100' not in os.listdir(scene_dir):
            im_dir = scene_dir
        else:
            print(f'Invalid scene {scene}')
            continue

        # Load images in descending order by names
        im_list = sorted(os.listdir(im_dir))[::-1]  
        print(f'Scene {scene} ims: {len(im_list)}')

        # Construct matching pairs
        pairs = [] 
        pair_keys = []
        for i, im1_name in enumerate(im_list):
            im1_key = os.path.splitext(os.path.basename(im1_name))[0]
            im1_path = os.path.join(im_dir, im1_name)
            for im2_name in im_list[i+1:]:
                im2_key = os.path.splitext(os.path.basename(im2_name))[0]                
                im2_path = os.path.join(im_dir, im2_name)                
                pairs.append((im1_path, im2_path))
                pair_keys.append(f'{im1_key}-{im2_key}')                

        # Save raw matches (under output_dir)
        raw_match_file = os.path.join(output_dir, f'{scene}-matches_raw.h5')
        if not skip_matching:
            match_pairs_with_keys_exporth5(matcher, pairs, pair_keys, raw_match_file)

        # Parse matches and keypoints
        parse_matches_and_keypoints(
            raw_match_file, results_dir, sc_thres, gv_thres, 
            qt_dthres=qt_dthres, qt_psize=qt_psize, qt_unique=qt_unique,
        )
           
            
def prepare_submission_local_features():
    # Optional TODO: prepare submission for local features
    pass
            
