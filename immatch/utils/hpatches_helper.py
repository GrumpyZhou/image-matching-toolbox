import os
import numpy as np
import glob
import time
import pydegensac

def eval_matches(p1s, p2s, homography):
    # Compute the reprojection errors from im1 to im2 
    # with the given the GT homography
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist

def eval_summary(results, thres=[1, 3, 5, 10], save_npy=None):
    np.set_printoptions(precision=2)
    summary = ''
    n_i = 52
    n_v = 56      
    i_err, v_err, stats = results
    seq_type, n_feats, n_matches = stats
    
    if save_npy:
        print(f'Save results to {save_npy}')
        np.save(save_npy, np.array(results, dtype=object))
        
    summary += '# Features: mean={:.0f} min={:d} max={:d}\n'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats))
    summary += '# Matches: a={:.0f}, i={:.0f}, v={:.0f}\n'.format(
                    np.sum(n_matches) / ((n_i + n_v) * 5), 
                    np.sum(n_matches[seq_type == 'i']) / (n_i * 5), 
                    np.sum(n_matches[seq_type == 'v']) / (n_v * 5)
                )

    thres = np.array(thres)
    ierr = np.array([i_err[th] / (n_i * 5) for th in thres ])
    verr = np.array([v_err[th] / (n_v * 5) for th in thres])
    aerr = np.array([(i_err[th] + v_err[th]) / ((n_i + n_v) * 5) for th in thres])
    summary += '{} px: a={}\ni={}\nv={}'.format(thres, aerr, ierr, verr)
    return summary

def eval_hpatches_matching(matcher, data_root, method='', thres=[1, 3, 5, 10],
                           lprint_=print, print_out=False, save_npy=None):
    """Evaluate a matcher on HPatches sequences for image matching task,
    following the protocol defined in D2Net.
    Args:
        - matcher: the matching function that inputs an image pair paths and 
                   outputs the matches and keypoints. 
        - data_root: the folder directory of HPatches dataset.
        - method: the description of the evaluated method.
        - thres: the mean matching accuracies are summarized/printed under the specified error thresholds in pixels. 
        - lprint: the printing function. If needed it can be implemented to outstream to a log file.
        - print_out: when set to True, per-pair results are printed during the evaluation.
        - save_npy: the path to save the result cache, by default nothing gets saved.
    """
    
    np.set_printoptions(precision=2)
    
    lprint_(f'>>Eval hpatches matching: method={method}')    
    seq_dirs = sorted(glob.glob('{}/*'.format(data_root)))
    n_feats = []
    n_matches = []
    seq_type = []
    match_time = []
    failed = 0

    thres_range = np.arange(1, 16)
    i_err = {thr: 0 for thr in thres_range}
    v_err = {thr: 0 for thr in thres_range}

    start_time = time.time()
    for seq_idx, seq_dir in enumerate(seq_dirs[::-1]): 
        sname = seq_dir.split('/')[-1]
        im1_path = os.path.join(seq_dir, '1.ppm')

        # Compare with other ims
        for im_idx in range(2, 7):
            im2_path = os.path.join(seq_dir, '{}.ppm'.format(im_idx))        
            homography = np.loadtxt(os.path.join(seq_dir, 'H_1_{}'.format(im_idx)))
            
            t0 = time.time()
            match_res = matcher(im1_path, im2_path)
            match_time.append(time.time() - t0)
            matches, kpts1, kpts2 = match_res[0:3]
            
            n_feats.append(kpts1.shape[0])
            n_feats.append(kpts2.shape[0])            
            n_matches.append(matches.shape[0])        
            seq_type.append(sname[0])

            if matches.shape[0] == 0:
                dist = np.array([float("inf")]) 
                failed += 1
            else:                   
                dist = eval_matches(matches[:, :2], matches[:, 2:], homography)
            if print_out:
                print('Scene {}, pair:1-{} matches:{} median_dist:{:.2f} <1px:{:.3f}'.format(sname, im_idx, len(dist), np.median(dist), np.mean(dist <= 1)))
            
            for thr in thres_range:
                if sname[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)
            
    total_time = time.time() - start_time
    
    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)
    results = i_err, v_err, [seq_type, n_feats, n_matches]
    
    lprint_(eval_summary(results, thres, save_npy=save_npy))
    lprint_('Finished, pairs={} failed={} total_time={:.2f}s match per pair={:.2f}s\n.'.format(len(match_time), failed, total_time, np.mean(match_time)))
    
    
    
def eval_hpatches_homography(matcher, data_root, method='', ransac_thres=[2], 
                             thres=[1, 3, 5, 10], lprint_=print, print_out=False):
    """Evaluate a matcher on HPatches sequences for homogray estimation.
    We follow CAPS protocol to measure the percentage of correctly estimated 
    homographies whose average corner error distance is below given thresholds.
    Args:
        - matcher: the matching function that inputs an image pair paths and 
                   outputs the matches and keypoints. 
        - data_root: the folder directory of HPatches dataset.
        - method: the description of the evaluated method.
        - ransac_thres: the set of ransac thresholds used by the solver to estimate homographies.
                        Results under each ransac threshold are printed per line.
        - thres: the correctness accuracies are printed under the specified error thresholds (in pixels).
        - lprint: the printing function. If needed it can be implemented to outstream to a log file.
        - print_out: when set to True, per-pair results are printed during the evaluation.
    """
    
    np.set_printoptions(precision=2)
    from PIL import Image
    
    corr_thres = thres
    seq_dirs = sorted(glob.glob('{}/*'.format(data_root)))
    lprint_(f'\n>>Eval hpatches homography: method={method} rthres={ransac_thres} thres={corr_thres} ')
    
    for rthres in ransac_thres:
        inlier_ratio = []
        num_matches = []
        correct_sa = []            
        correct_si = []
        correct_sv = []
        match_time = []
        start_time = time.time()        
        for seq_idx, seq_dir in enumerate(seq_dirs[::-1]): 
            sname = seq_dir.split('/')[-1]
            im1_path = os.path.join(seq_dir, '1.ppm')

            # Compare with other ims
            for im_idx in range(2, 7):
                im2_path = os.path.join(seq_dir, '{}.ppm'.format(im_idx))
                H_gt = np.loadtxt(os.path.join(seq_dir, 'H_1_{}'.format(im_idx)))
                                
                try:
                    t0 = time.time()
                    match_res = matcher(im1_path, im2_path)
                    match_time.append(time.time() - t0)
                    matches = match_res[0]
                    p1s = matches[:, :2]        
                    p2s = matches[:, 2:]
                except:
                    p1s = p2s = []
                num_matches.append(len(p1s))            


                # Estimate the homography between the matches using RANSAC
                try:
                    H_pred, inliers = pydegensac.findHomography(p1s, p2s, rthres)
                    #H_pred, inliers = cv2.findHomography(p1s, p2s, cv2.RANSAC)
                except:
                    H_pred = None

                if H_pred is None:
                    correctness = [0.0 for cthr in corr_thres]
                    irat = 0
                else:
                    im = Image.open(im1_path)
                    w, h = im.size
                    corners = np.array([[0, 0, 1],
                                        [0, w - 1, 1],
                                        [h - 1, 0, 1],
                                        [h - 1, w - 1, 1]])
                    real_warped_corners = np.dot(corners, np.transpose(H_gt))
                    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                    warped_corners = np.dot(corners, np.transpose(H_pred))
                    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
                    correctness = [float(mean_dist <= cthr) for cthr in corr_thres]
                    irat = np.mean(inliers) 
                if print_out:
                    print(im_idx, len(p1s), correctness, np.sum(inliers))
                correct_sa.append(correctness)
                inlier_ratio.append(irat)
                if sname[0] == 'i':
                    correct_si.append(correctness)
                if sname[0] == 'v':
                    correct_sv.append(correctness)

        lprint_('ransac={} N={:.1f} irate={:.2f}  match_time={:.2f}s ' \
                'correct:a={} i={} v={}'.format(rthres,
                                                np.mean(num_matches), 
                                                np.mean(inlier_ratio),
                                                np.mean(match_time),
                                                np.mean(correct_sa, axis=0), 
                                                np.mean(correct_si, axis=0), 
                                                np.mean(correct_sv, axis=0)))

def eval_hpatches(matcher, data_root, method='', ransac_thres=2, 
                  thres=[1, 3, 5, 10], lprint_=print, print_out=False, save_npy=None,):
    """Evaluate a matcher on HPatches sequences for homogray estimation.
    We follow CAPS protocol to measure the percentage of correctly estimated 
    homographies whose average corner error distance is below given thresholds.
    Args:
        - matcher: the matching function that inputs an image pair paths and 
                   outputs the matches and keypoints. 
        - data_root: the folder directory of HPatches dataset.
        - method: the description of the evaluated method.
        - ransac_thres: the set of ransac thresholds used by the solver to estimate homographies.
                        Results under each ransac threshold are printed per line.
        - thres: the correctness accuracies are printed under the specified error thresholds (in pixels).
        - lprint: the printing function. If needed it can be implemented to outstream to a log file.
        - print_out: when set to True, per-pair results are printed during the evaluation.
    """
    
    np.set_printoptions(precision=2)
    from PIL import Image
    
    corr_thres = thres
    seq_dirs = sorted(glob.glob('{}/*'.format(data_root)))
    lprint_(f'\n>>Eval hpatches: method={method} rthres={ransac_thres} thres={corr_thres} ')
    
    # Matching
    thres_range = np.arange(1, 16)
    i_err = {thr: 0 for thr in thres_range}
    v_err = {thr: 0 for thr in thres_range}
    n_feats = []
    n_matches = []
    seq_type = []
    
    # Homography    
    inlier_ratio = []
    correct_sa = []            
    correct_si = []
    correct_sv = []
    
    failed = 0
    match_time = []
    start_time = time.time()        
    for seq_idx, seq_dir in enumerate(seq_dirs[::-1]): 
        sname = seq_dir.split('/')[-1]
        im1_path = os.path.join(seq_dir, '1.ppm')

        # Compare with other ims
        for im_idx in range(2, 7):
            im2_path = os.path.join(seq_dir, '{}.ppm'.format(im_idx))
            H_gt = np.loadtxt(os.path.join(seq_dir, 'H_1_{}'.format(im_idx)))
            match_res = matcher(im1_path, im2_path)
            try:
                t0 = time.time()
                match_res = matcher(im1_path, im2_path)
                match_time.append(time.time() - t0)
                matches, p1s, p2s = match_res[0:3]
            except:
                p1s = p2s = matches = []
            n_feats.append(len(p1s))
            n_feats.append(len(p2s))            
            n_matches.append(matches.shape[0])
            seq_type.append(sname[0])
            
            if len(matches) == 0:
                dist = np.array([float("inf")]) 
                failed += 1
            else:                   
                dist = eval_matches(matches[:, :2], matches[:, 2:], H_gt)
            if print_out:
                print('Scene {}, pair:1-{} matches:{} median_dist:{:.2f} <1px:{:.3f}'.format(sname, im_idx, len(dist), np.median(dist), np.mean(dist <= 1)))
                
            for thr in thres_range:
                if sname[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)
                    

            # Estimate the homography between the matches using RANSAC
            try:
                H_pred, inliers = pydegensac.findHomography(matches[:, :2], matches[:, 2:4], ransac_thres)
                #H_pred, inliers = cv2.findHomography(p1s, p2s, cv2.RANSAC)
            except:
                H_pred = None

            if H_pred is None:
                correctness = [0.0 for cthr in corr_thres]
                irat = 0
            else:
                im = Image.open(im1_path)
                w, h = im.size
                corners = np.array([[0, 0, 1],
                                    [0, w - 1, 1],
                                    [h - 1, 0, 1],
                                    [h - 1, w - 1, 1]])
                real_warped_corners = np.dot(corners, np.transpose(H_gt))
                real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
                correctness = [float(mean_dist <= cthr) for cthr in corr_thres]
                irat = np.mean(inliers)
            correct_sa.append(correctness)
            inlier_ratio.append(irat)
            if sname[0] == 'i':
                correct_si.append(correctness)
            if sname[0] == 'v':
                correct_sv.append(correctness)
                
    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)
    results = i_err, v_err, [seq_type, n_feats, n_matches]
    lprint_(eval_summary(results, thres, save_npy=save_npy))
    lprint_('pairs={} failed={}\n'.format(len(match_time), failed,  np.mean(match_time)))
    lprint_('Finished, ransac={} N={:.1f} irate={:.2f}  match_time={:.2f}s ' \
            'correct:a={} i={} v={}'.format(ransac_thres,
                                            np.mean(n_matches), 
                                            np.mean(inlier_ratio),
                                            np.mean(match_time),
                                            np.mean(correct_sa, axis=0), 
                                            np.mean(correct_si, axis=0), 
                                            np.mean(correct_sv, axis=0)))        
 

