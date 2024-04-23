import argparse
from argparse import Namespace
import os
from pathlib import Path

from immatch.utils.model_helper import init_model
from immatch.utils.localize_sfm_helper import *


def generate_covis_db_pairs(args):
    from immatch.utils.colmap.data_parsing import (
        covis_pairs_from_nvm,
        covis_pairs_from_reference_model,
    )

    if args.benchmark_name in ["aachen"]:
        nvm_path = Path(args.dataset_dir) / "3D-models/aachen_cvpr2018_db.nvm"
        covis_pairs_from_nvm(nvm_path, args.pair_dir, topk=args.topk)

    if args.benchmark_name in ["aachen_v1.1"]:
        model_dir = Path(args.dataset_dir) / "3D-models/aachen_v_1_1"
        covis_pairs_from_reference_model(model_dir, args.pair_dir, topk=args.topk)


def eval_aachen(args):
    # Initialize model
    model, model_conf = init_model(args.config, args.benchmark_name)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)

    # Merge args
    args = Namespace(**vars(args), **model_conf)
    args.model_tag = model.name
    args.conf_tag = f"im{model.imsize}"
    if args.prefix:
        args.conf_tag += f".{args.prefix}"

    # Load db & query pairs
    pair_list = load_pairs(args)
    if not pair_list:
        return

    # Initialize experiment paths
    args.dataset_dir = Path(args.dataset_dir)
    args.im_dir = args.dataset_dir / "images/images_upright"
    args = init_paths(args)
    print(args)

    # Match pairs
    if not args.skip_match:
        match_pairs_exporth5(pair_list, matcher, args.im_dir, args.output_dir)

    # Extract keypoints
    process_matches_and_keypoints_exporth5(
        pair_list,
        args.output_dir,
        args.result_dir,
        qt_psize=args.qt_psize,
        qt_dthres=args.qt_dthres,
        sc_thres=args.sc_thres,
    )

    # Localization
    init_empty_sfm(args)
    reconstruct_database_pairs(args)
    localize_queries(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localize Aachen Day-Night")
    parser.add_argument("--gpu", "-gpu", type=str, default="0")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--colmap", type=str, required=True)
    parser.add_argument("--skip_match", action="store_true")
    parser.add_argument(
        "--dataset_dir", type=str, default="data/datasets/AachenDayNight"
    )
    parser.add_argument("--pair_dir", type=str, default="data/pairs")
    parser.add_argument(
        "--benchmark_name",
        type=str,
        choices=["aachen", "aachen_v1.1"],
        default="aachen",
    )

    # Turn on to only generate covis db pairs
    parser.add_argument("--generate_covis_db_pairs", action="store_true")
    parser.add_argument("--topk", type=int, default=20)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Generate covis pairs
    if args.generate_covis_db_pairs:
        generate_covis_db_pairs(args)
    else:
        eval_aachen(args)
