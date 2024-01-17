from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm
import argparse
from argparse import Namespace
import cv2
from pathlib import Path

from immatch.utils.model_helper import init_model
import immatch.utils.metrics as M

def compute_relapose_aspan(kpts0, kpts1, K0, K1, pix_thres=0.5, conf=0.99999):
    """ Original code from ASpanFormer repo:
        https://github.com/apple/ml-aspanformer/blob/main/src/utils/metrics.py
    """

    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = pix_thres / np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def load_scannet_pairs_npz(intrinsic_path, npz_path, data_root):
    # Load intrinsics
    intrinsics = dict(np.load(intrinsic_path))

    # Collect pairs
    pairs = []
    scene_info = dict(np.load(npz_path))
    for scene_id, scene_sub, stem_name_1, stem_name_2 in tqdm(scene_info['name']):
        scene_name = f'scene{scene_id:04d}_{scene_sub:02d}'

        im1 = f"{scene_name}/color/{stem_name_1}.jpg"
        im2 = f"{scene_name}/color/{stem_name_2}.jpg"
        K1 = K2 = np.array(intrinsics[scene_name].copy(), dtype=np.float32).reshape(3, 3)

        # Compute relative pose
        T1 = np.linalg.inv(np.loadtxt(f"{data_root}/{scene_name}/pose/{stem_name_1}.txt", delimiter=' '))
        T2 = np.linalg.inv(np.loadtxt(f"{data_root}/{scene_name}/pose/{stem_name_2}.txt", delimiter=' '))
        T12 = np.matmul(T2, np.linalg.inv(T1))
        pairs.append(Namespace(
            im1=im1, im2=im2, K1=K1, K2=K2, t=T12[:3, 3], R=T12[:3, :3]
        ))
    print(f"Loaded {len(pairs)} pairs.")
    return pairs

def load_megadepth_pairs_npz(npz_root, npz_list):
    with open(npz_list, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
    print(f"Parse {len(npz_names)} npz from {npz_list}.")

    pairs = []
    for name in npz_names:
        scene_info = np.load(f"{npz_root}/{name}.npz", allow_pickle=True)

        # Collect pairs
        for pair_info in scene_info['pair_infos']:
            (id1, id2), overlap, _ = pair_info
            im1 = scene_info['image_paths'][id1].replace('Undistorted_SfM/', '')
            im2 = scene_info['image_paths'][id2].replace('Undistorted_SfM/', '')                        
            K1 = scene_info['intrinsics'][id1].astype(np.float32)
            K2 = scene_info['intrinsics'][id2].astype(np.float32)

            # Compute relative pose
            T1 = scene_info['poses'][id1]
            T2 = scene_info['poses'][id2]
            T12 = np.matmul(T2, np.linalg.inv(T1))
            pairs.append(Namespace(
                im1=im1, im2=im2, overlap=overlap, 
                K1=K1, K2=K2, t=T12[:3, 3], R=T12[:3, :3]
            ))
    print(f"Loaded {len(pairs)} pairs.")
    return pairs

def eval_relapose(
    matcher,
    data_root,
    pairs,
    method='',
    ransac_thres=0.5,
    thresholds=[1, 3, 5, 10, 20],
    print_out=False,
    debug=False,
):
    statis = defaultdict(list)
    np.set_printoptions(precision=2)

    # Eval on pairs
    print(f">>> Start eval on Megadepth: method={method} rthres={ransac_thres} ... \n")
    for i, pair in tqdm(enumerate(pairs), smoothing=.1, total=len(pairs)):
        if debug and i > 10:
            break

        K1 = pair.K1
        K2 = pair.K2
        t_gt = pair.t
        R_gt = pair.R
        im1 = str(data_root / pair.im1)
        im2 = str(data_root / pair.im2)
        match_res = matcher(im1, im2)
        matches, pts1, pts2 = match_res[0:3]

        # Compute pose errors
        ret = compute_relapose_aspan(
            pts1, pts2, K1, K2, pix_thres=ransac_thres
        )

        if ret is None:
            statis['failed'].append(i)
            statis['R_errs'].append(np.inf)
            statis['t_errs'].append(np.inf)
            statis['inliers'].append(np.array([]).astype(np.bool_))
        else:
            R, t, inliers = ret
            R_err, t_err = M.cal_relapose_error(R, R_gt, t, t_gt)
            statis['R_errs'].append(R_err)
            statis['t_errs'].append(t_err)
            statis['inliers'].append(inliers.sum() / len(pts1))
            if print_out:
                print(f"#M={len(matches)} R={R_err:.3f}, t={t_err:.3f}")

    print(f"Total samples: {len(pairs)} Failed:{len(statis['failed'])}.")
    pose_auc = M.cal_relapose_auc(statis, thresholds=thresholds)
    return pose_auc

def eval_megadepth_relapose(
    root_dir,
    method,
    benchmark='megadepth',
    ransac_thres=0.5,
    print_out=False,
    debug=False,
):

    # Init paths
    npz_root = root_dir / 'third_party/aspanformer/assets/megadepth_test_1500_scene_info'
    npz_list = root_dir / 'third_party/aspanformer/assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt'
    data_root = root_dir / 'data/datasets/MegaDepth_undistort'

    # Load pairs
    pairs = load_megadepth_pairs_npz(npz_root, npz_list)
        
    # Init model
    model, config = init_model(method, benchmark, root_dir=root_dir)    
    matcher = lambda im1, im2: model.match_pairs(im1, im2)

    # Eval
    eval_relapose(
        matcher, 
        data_root,
        pairs,
        model.name,
        print_out=print_out,
        debug=debug,
    )

def eval_scannet_relapose(
    root_dir,
    method,
    benchmark='scannet',
    ransac_thres=0.5,
    print_out=False,
    debug=False,
):

    # Init paths
    npz_path = root_dir / 'third_party/aspanformer/assets/scannet_test_1500/test.npz'
    intrinsic_path = root_dir / 'third_party/aspanformer/assets/scannet_test_1500/intrinsics.npz'
    data_root = root_dir / 'data/datasets/scannet/test'

    # Load pairs
    pairs = load_scannet_pairs_npz(intrinsic_path, npz_path, data_root)

    # Init model
    model, config = init_model(method, benchmark, root_dir=root_dir)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)

    # Eval
    eval_relapose(
        matcher,
        data_root,
        pairs,
        model.name,
        print_out=print_out,
        debug=debug,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Relative Pose')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--benchmark', type=str, required=True, choices=['scannet', 'megadepth'])
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--ransac_thres', type=float, default=0.5)
    parser.add_argument('--print_out', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    eval(f"eval_{args.benchmark}_relapose")(
        Path(args.root_dir),
        args.config,
        benchmark=args.benchmark,
        ransac_thres=args.ransac_thres,
        print_out=args.print_out,
        debug=args.debug,
    )
