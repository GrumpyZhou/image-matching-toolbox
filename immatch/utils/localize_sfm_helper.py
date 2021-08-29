import numpy as np
import time
import logging
from tqdm import tqdm
import h5py
import os
from pathlib import Path

def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))

def load_pairs(args):
    args.pair_dir = Path(args.pair_dir, args.benchmark_name)
    args.db_pairs_path = args.pair_dir / args.pairs[0]
    args.query_pairs_path = args.pair_dir / args.pairs[1]
    if not args.db_pairs_path.exists():
        print(f'{args.db_pairs_path} does not exist!!')
        return None
    if not args.db_pairs_path.exists():
        print(f'{args.query_pairs_path} does not exist!!')
        return None
    
    # Load pairs
    print(f'Pair list: {args.pairs}')
    db_pairs = []
    query_pairs = []
    with open(args.db_pairs_path) as f:
        db_pairs += f.readlines()
    with open(args.query_pairs_path) as f:
        query_pairs += f.readlines()    
    print(f'Loaded pairs db:{len(db_pairs)} query:{len(query_pairs)}')
    pair_list = db_pairs + query_pairs
    return pair_list

def init_paths(args):
    # Define experiment tags
    if args.qt_psize > 0:
        args.qt_tag = f'qt{args.qt_psize}d{args.qt_dthres}sc{args.sc_thres}'
        if args.qt_unique:
                args.qt_tag += 'uni'
    else:
        args.qt_tag = f'sc{args.sc_thres}'    
    
    db_pair_tag = args.pairs[0].replace('db-pairs-', '').replace('.txt', '')
    query_pair_tag = args.pairs[1].replace('pairs-query-', '').replace('.txt', '')
    args.pair_tag = db_pair_tag + query_pair_tag
    
    # Output paths 
    output_dir = Path('outputs', args.benchmark_name, args.model_tag, args.conf_tag)
    if not output_dir.exists():
        os.makedirs(output_dir)
    args.output_dir = output_dir
    args.empty_sfm = Path('outputs', args.benchmark_name, 'empty_sfm')
    
    # Result dir for sfm models and localization results
    result_dir = output_dir / f'{args.pair_tag}_{args.qt_tag}'
    if not result_dir.exists():
        os.makedirs(result_dir)
    logging.info(f'Result folder: {result_dir}')
    args.result_dir = result_dir
    args.result_sfm = result_dir / 'sfm_final'
    return args

def init_empty_sfm(args):
    from immatch.utils.colmap.data_parsing import (
        create_empty_model_from_nvm_and_database, create_empty_model_from_reference_model
    )
    
    if args.empty_sfm.exists():
        logging.info('Empty sfm existed.')
        return
    dataset_dir = args.dataset_dir
    
    if args.benchmark_name == 'robotcar':
        create_empty_model_from_nvm_and_database(
            dataset_dir / '3D-models/all-merged/all.nvm',
            dataset_dir / '3D-models/overcast-reference.db',
            args.empty_sfm
        )
    elif args.benchmark_name in ['aachen']: 
        # Original Aachen  
        logging.info('Init empty sfm from nvm for aachen...')
        create_empty_model_from_nvm_and_database(
            dataset_dir  / '3D-models/aachen_cvpr2018_db.nvm',
            dataset_dir  / 'database.db',
            args.empty_sfm,
            dataset_dir / '3D-models/database_intrinsics.txt',    
        )
    elif args.benchmark_name in ['aachen_v1.1']:
        # Aachen v1.1
        logging.info('Init empty sfm from bins for aachenv1.1...')                
        create_empty_model_from_reference_model(
            dataset_dir / '3D-models/aachen_v_1_1',
            args.empty_sfm
        )
    else:
        logging.error(f'Invalid benchmark {args.benchmark_name}!!')
    
def reconstruct_database_pairs(args):
    from third_party.hloc.hloc import triangulation
    
    # Reconstruct database pairs
    if (args.result_sfm / 'model' / 'images.bin').exists():
        logging.info('Reconstruction existed.')
        return    
    logging.info('\nReconstructing database images....')
    triangulation.main(
        args.result_sfm,
        args.empty_sfm,
        args.im_dir,
        args.db_pairs_path,
        args.result_dir / 'keypoints.h5',
        args.result_dir /'matches.h5',
        colmap_path=args.colmap)
    logging.info('Finished reconstruction.')
    
def localize_queries(args):
    from third_party.hloc.hloc import localize_sfm
    
    # Localize query pairs
    for ransac_thresh in args.ransac_thres:
        localize_txt = f'{args.model_tag}_{args.conf_tag}.{args.pair_tag}.{args.qt_tag}.r{ransac_thresh}.txt'
        if args.covis_cluster:
            localize_txt = localize_txt.replace('txt', 'covis.txt')
        logging.info(f'Localize queries...')
        print('>>>>', localize_txt)        
        localize_sfm.main(
            args.result_sfm / 'model',
            args.dataset_dir / 'queries/*_queries_with_intrinsics.txt',
            args.query_pairs_path,
            args.result_dir / 'keypoints.h5',
            args.result_dir / 'matches.h5',
            args.result_dir / localize_txt,
            ransac_thresh=ransac_thresh,
            covisibility_clustering=args.covis_cluster,
            changed_format=True,
            benchmark=args.benchmark_name
        )

def get_grouped_ids(array):
    # Group array indices based on its values
    # all duplicates are grouped as a set
    idx_sort = np.argsort(array)
    sorted_array = array[idx_sort]
    vals, ids, count = np.unique(sorted_array, return_counts=True, return_index=True)
    res = np.split(idx_sort, ids[1:])
    return res

def get_unique_matches_ids(match_ids, scores):
    if len(match_ids.shape) == 1:
        return [0]

    k1s = match_ids[:, 0]
    k2s = match_ids[:, 1]
    isets1 = get_grouped_ids(k1s)
    isets2 = get_grouped_ids(k2s)
    
    uid1s = []
    for ids in isets1:
        if len(ids) == 1:
            uid1s.append(ids[0])  # Unique
        else:        
            uid1s.append(ids[scores[ids].argmax()])
    uid2s = []
    for ids in isets2:
        if len(ids) == 1:
            uid2s.append(ids[0])  # Unique
        else:    
            uid2s.append(ids[scores[ids].argmax()])            
    uids = list(set(uid1s).intersection(uid2s))    
    return uids  

def quantize_keypoints(fpts, kp_data, psize=48, dthres=4):
    """Keypoints quantization algorithm.
    The image is divided into cells, where each cell represents a psize*psize local patch.
    Each input point has its linked coarse point (patch region).
    For all points inside a patch region, those points with distances smaller than the given 
    threshold will be merged and represented by their mean pixel.
    
    Args:
        - fpts: the set of input keypoint coordinates, shape (N, 2)
        - kp_data: the keypoint data dict linked to an image
        - psize: the size to divide an image into multiple patch regions
        - dthres: the distance threshold (in pixels) for merging points within the same patch region        
    Return:
        - fpt_ids: the keypoint ids of the input points
    """
    
    # kp_data: {'kps':[], 'kp_means': kp_dict}
    fpt_ids = []
    cpts = fpts // psize * psize   # Point coordinates (x, y)      
    for cpt, fpt in zip(cpts, fpts):
        cpt = tuple(cpt) 
        kps = kp_data['kps']
        kp_dict = kp_data['kp_means']  # {cpt : {'means':[], 'kids':[]}}
        if cpt not in kp_dict:
            kid = len(kps)
            kps.append(fpt)  # Insert another keypoint
            kp_dict[cpt] = {'means':[fpt], 'kids':[kid]}  # Init 1st center        
        else:
            kids = kp_dict[cpt]['kids']
            centers = kp_dict[cpt]['means']  # N, 2 
            dist = np.linalg.norm(fpt - np.array(centers), axis=1)
            cid = np.argmin(dist)        
            if dist[cid] < dthres:
                centers[cid] = (centers[cid] + fpt) / 2   # Update center
                kid = kids[cid]
                kps[kid] = centers[cid]  # Update key point value
            else:
                kid = len(kps)
                kps.append(fpt)  # Insert another keypoint
                centers.append(fpt)  # Insert as a new center
                kids.append(kid)
        fpt_ids.append(kid)    
    return fpt_ids

def compute_keypoints(pts, kp_data):
    kps = kp_data['kps']
    kp_dict = kp_data['kpids']
    pt_ids = []
    for pt in pts:
        key = tuple(pt)        
        if key not in kp_dict:
            kid = len(kps)   # 0-based, the next inserting index            
            kps.append(pt)
            kp_dict[key] = kid
        else:
            kid = kp_dict[key]            
        pt_ids.append(kid)
    return pt_ids

def match_pairs_with_keys_exporth5(matcher, pairs, pair_keys, match_file, debug=False):
    # Pairwise matching
    num_matches = []
    match_times = []
    start_time = time.time()

    with h5py.File(match_file, 'a') as fmatch:
        matched = list(fmatch.keys())
        print(f'\nLoad match file, existing matches {len(matched)}')
        print(f'Start matching, total {len(pairs)} pairs...')
        for pair, key in tqdm(zip(pairs, pair_keys), total=len(pairs), smoothing=.1):
            im1_path, im2_path = pair
            if key in matched:
                num_matches.append(len(fmatch[key]['matches']))
                continue

            try:
                t0 = time.time()
                match_res = matcher(im1_path, im2_path)
                match_times.append(time.time() - t0)
            except:
                print(f'##Failed matching on {key}')
                continue

            matches = match_res[0]
            scores = match_res[-1]
            N = len(matches)
            num_matches.append(N)

            # Add print for easy debugging
            if debug:
                print(f'{pair} matches: {N}')

            # Save matches
            grp = fmatch.create_group(key)
            grp.create_dataset('matches', data=matches)
            grp.create_dataset('scores', data=scores)
        total_time = time.time() - start_time
        mean_time = np.mean(match_times) if len(match_times) > 0 else 0.0
        print(f'Finished matched pairs: {len(fmatch)} num_matches:{np.mean(num_matches):.2f} '
              f'match_time/pair:{mean_time:.2f}s time:{total_time:.2f}s.')

def match_pairs_exporth5(pair_list, matcher, im_dir, output_dir, debug=False):
    # Construct pairs and pair keys
    pairs = []
    pair_keys = []
    pair_keys_set = set()
    for pair_line in tqdm(pair_list, smoothing=.1):
        name0, name1 = pair_line.split()
        key = names_to_pair(name0, name1)
        key_inv = names_to_pair(name1, name0)
        if key_inv in pair_keys_set:
            continue

        pair_keys.append(key)
        pair_keys_set.add(key)
        pair = (str(im_dir / name0), str(im_dir / name1))
        pairs.append(pair)

    match_file = output_dir/'matches_raw.h5'
    match_pairs_with_keys_exporth5(matcher, pairs, pair_keys, match_file, debug=debug)

def process_matches_and_keypoints_exporth5(pair_list, output_dir, result_dir,    
                                           sc_thres=0.25, qt_dthres=4, qt_psize=48, 
                                           qt_unique=True):
    """
    This function should be executed after running match_pairs_exporth5(), which save the 
    precomputed the raw matches and their scores in a hdf5.
    This function first filters out the less confident matches based on the given score threshold.
    Then the keypoints are computed accordingly from the filtered matches by finding the unique set
    of points that have been matched for each image.
    Afterwards, matches are represented by pairs of keypoint ids.
    We further provide an option to quantize keypoints, where keypoints are closer than 
    a given distance will be merged into one.
    For the methods that directly obtain pixel-level matches without having an explicit keypoint 
    detection stage, such quantization could be helpful or necessary to enable proper stability
    for the triangulation step of a localization pipeline. 
    Notice, such quantization sacrifices the pixel-wise accuracy, so the methods with 
    specific keypoint detection (SuperPoint, D2Net, ..etc) or the methods that produce 
    patch-level matches (NCNet, SparseNCNet) should not use it.
    
    Args:
        - pair_list: list of pair strings
        - output_dir: the directory where matches_raw.h5 is saved
        - result_dir: the output directory to save processed matches and keypoints
        - sc_thres: the score threshold to filter less confident matches
        - qt_dthres, qt_psize: args for quantization, set qt_psize as 0 can skip the quantization.
        - qt_unique: flag to maintain the uniqueness of matches after quantization

    Nothing is returned. The processed keypoints are saved as keypoints.h5 and matches (represented by the 
    keypoints ids) are saved as matches.h5 in the result directory.
    """
    # Select matches from the raw matches and quantize keypoints   
    if (result_dir/'keypoints.h5').exists():
        logging.info('Result matches and keypoints already existed, skip')
        return
        
    match_file = h5py.File(output_dir/'matches_raw.h5', 'r')
    all_kp_data = {}
    logging.info('Start parse matches and quantize keypoints ...')
    with h5py.File(result_dir/'matches.h5', 'w') as res_fmatch:
        for pair in tqdm(pair_list, smoothing=.1):
            name0, name1 = pair.split()
            pair = names_to_pair(name0, name1)

            if pair not in match_file:
                continue            

            matches = match_file[pair]['matches'].__array__()
            scores = match_file[pair]['scores'].__array__()
            valid = np.where(scores >= sc_thres)[0]
            matches = matches[valid]
            scores = scores[valid]
            
            # Compute match ids and quantize keypoints
            match_ids = matches_to_keypoint_ids(
                matches, scores, name0, name1, all_kp_data,
                qt_dthres, qt_psize, qt_unique
            )

            # Save matches
            grp = res_fmatch.create_group(pair)
            grp.create_dataset('matches0', data=match_ids)
        num_pairs = len(res_fmatch.keys())
        
    # Save keypoints
    with h5py.File(result_dir/'keypoints.h5', 'w') as res_fkp:    
        logging.info(f'Save keypoints from {len(all_kp_data)} images...')
        for name in tqdm(all_kp_data, smoothing=.1):
            kps = np.array(all_kp_data[name]['kps'], dtype=np.float32)
            kgrp = res_fkp.create_group(name)
            kgrp.create_dataset('keypoints', data=kps)
    logging.info(f'Finished quantization, match pairs:{num_pairs}')
    match_file.close()
    
def matches_to_keypoint_ids(matches, scores, name0, name1, all_kp_data,
                            qt_dthres=-1, qt_psize=-1, qt_unique=True):
    if len(matches) == 0:
        return np.empty([0, 2], dtype=np.int32)

    if qt_psize > 0 and qt_dthres > 0:
        # Compute keypoints jointly with quantization
        if name0 not in all_kp_data:
            all_kp_data[name0] = {'kps':[], 'kp_means':{}}
        if name1 not in all_kp_data:
            all_kp_data[name1] = {'kps':[], 'kp_means':{}}

        id1s = quantize_keypoints(matches[:, 0:2], all_kp_data[name0],
                                  psize=qt_psize, dthres=qt_dthres)
        id2s = quantize_keypoints(matches[:, 2:4], all_kp_data[name1],
                                  psize=qt_psize, dthres=qt_dthres)
        match_ids = np.dstack([id1s, id2s]).squeeze() # N, 2

        # Remove n-to-1 matches after quantization
        if qt_unique and len(match_ids) > 1:
            uids = get_unique_matches_ids(match_ids, scores)
            match_ids = match_ids[uids]
            uids = get_unique_matches_ids(match_ids, scores)
            match_ids = match_ids[uids]
    else:
        # Compute keypoints without quantization
        if name0 not in all_kp_data:
            all_kp_data[name0] = {'kps':[], 'kpids':{}}
        if name1 not in all_kp_data:
            all_kp_data[name1] = {'kps':[], 'kpids':{}}

        id1s = compute_keypoints(matches[:, 0:2], all_kp_data[name0])
        id2s = compute_keypoints(matches[:, 2:4], all_kp_data[name1])
        match_ids = np.dstack([id1s, id2s]).squeeze()

    return match_ids

