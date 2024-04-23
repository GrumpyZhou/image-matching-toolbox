import torch
import numpy as np


def check_data_hist(data_list, bins, tag="", return_hist=False):
    if not data_list:
        if return_hist:
            return "", None
        return ""
    hists = []
    means = []
    Ns = []
    for data in data_list:
        N = len(data)
        Ns.append(N)
        if N == 0:
            continue
        counts = np.histogram(data, bins)[0]
        hists.append(counts / N)
        means.append(np.mean(data))

    hist_print = f"{tag} Sample/N(mean/max/min)={len(data_list)}/{np.mean(Ns):.0f}/{np.max(Ns):.0f}/{np.min(Ns):.0f}\n"
    hist_print += f"Ratios(%): mean={np.mean(means):.2f}"
    mean_hists = np.mean(hists, axis=0)
    for val, low, high in zip(mean_hists, bins[0:-1], bins[1::]):
        hist_print += " [{},{})={:.2f}".format(low, high, 100 * val)
    if return_hist:
        return mean_hists, hist_print
    return hist_print


def cal_relapose_auc(statis, thresholds=[5, 10, 20]):
    min_pose_err = np.maximum(np.array(statis["R_errs"]), np.array(statis["t_errs"]))
    auc = cal_error_auc(min_pose_err, thresholds)
    print(f"RelaPose AUC@{'/'.join(map(str, thresholds))}deg: {auc}%")
    return auc


def cal_error_auc(errors, thresholds):
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    N = len(errors)
    errors = np.append([0.0], np.sort(errors))
    recalls = np.arange(N + 1) / N
    aucs = []
    for thres in thresholds:
        last_index = np.searchsorted(errors, thres)
        rcs_ = np.append(recalls[:last_index], recalls[last_index - 1])
        errs_ = np.append(errors[:last_index], thres)
        aucs.append(np.trapz(rcs_, x=errs_) / thres)
    return 100 * np.array(aucs)


def cal_abspose_error(R, R_gt, t, t_gt):
    if isinstance(R_gt, np.ndarray):
        R_gt = torch.from_numpy(R_gt).to(torch.float32)
    if isinstance(t_gt, np.ndarray):
        t_gt = torch.from_numpy(t_gt).to(torch.float32)
    R_err = (
        torch.clip(0.5 * (torch.sum(R_gt * R_gt.new_tensor(R)) - 1), -1, 1).acos()
        * 180.0
        / pi
    )
    t_err = torch.norm(t_gt.new_tensor(t) - t_gt)
    return R_err, t_err


def cal_rot_error(R, R_gt):
    d = np.clip((np.trace(R.T.dot(R_gt)) - 1) / 2, -1.0, 1.0)
    err = np.rad2deg(np.arccos(d))
    return err


def cal_vec_angular_error(t, t_gt):
    norm = np.linalg.norm(t_gt) * np.linalg.norm(t)
    d = np.clip(np.dot(t, t_gt) / norm, -1.0, 1.0)
    err = np.rad2deg(np.arccos(d))

    # This is what Aspanformer is using ...
    # t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    return err


def cal_relapose_error(R, R_gt, t, t_gt):
    t_err = cal_vec_angular_error(t, t_gt)
    R_err = cal_rot_error(R, R_gt)
    return R_err, t_err


def cal_reproj_dists_H(p1s, p2s, homography):
    """Compute the reprojection errors using the GT homography"""

    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist
