from tools.esm_utils import aligned_distances_from_file
import argparse
import numpy as np
import torch
import json
import os
import tqdm
from tools import smart_rmsd


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_path",
        type=str,
        default=None,
        help="Path to the prediction file with tag format place",
    )
    parser.add_argument(
        "--tag_cluster_mapping",
        type=str,
        required=True,
        help="Path to tag cluster mapping file",
    )
    parser.add_argument(
        "--sequence_dict", type=str, required=True, help="Path to sequence dictionary"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--structure_dir",
        type=str,
        required=True,
        help="Directory containing structure files",
    )
    parser.add_argument(
        "--which_to_save",
        type=str,
        required=True,
        choices=["close", "far", "both"],
        help="Which distance to save",
    )

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.tag_cluster_mapping, "r") as f:
        tag_cluster_mapping = json.load(f)

    with open(args.sequence_dict, "r") as f:
        sequence_dict = json.load(f)

    if args.pred_path is not None:
        close_dir = os.path.join(args.output_dir, "close")
        far_dir = os.path.join(args.output_dir, "far")
        os.makedirs(close_dir, exist_ok=True)
        os.makedirs(far_dir, exist_ok=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        close_dir, far_dir = args.output_dir, args.output_dir

    all_tag_orders = {}
    for tag, cluster in tqdm.tqdm(tag_cluster_mapping.items()):
        sequence = sequence_dict[tag]
        gt_paths = [
            os.path.join(args.structure_dir, f"{pdb_id[:4]}.cif") for pdb_id in cluster
        ]

        if args.pred_path is not None:
            pred_path = args.pred_path.format(tag=tag)
            rmsds = [
                smart_rmsd(
                    pred_path, gt_path, chain1="A", chain2=pdb_id[5:], print_stuff=False
                )
                for (gt_path, pdb_id) in zip(gt_paths, cluster)
            ]

            close, far = (0, 1) if rmsds[0] < rmsds[1] else (1, 0)
        else:
            close, far = 0, 1

        close_tag, far_tag = cluster[close], cluster[far]
        all_tag_orders[tag] = [close_tag, far_tag]
        if args.which_to_save in ["close", "both"]:
            distances, mask = aligned_distances_from_file(
                sequence, gt_paths[close], close_tag[5:]
            )
            close_dist_path = os.path.join(close_dir, f"{tag}_dist.npy")
            np.save(close_dist_path, distances)
            close_mask_path = os.path.join(close_dir, f"{tag}_mask.npy")
            np.save(close_mask_path, mask)
        if args.which_to_save in ["far", "both"]:
            distances, mask = aligned_distances_from_file(
                sequence, gt_paths[far], far_tag[5:]
            )
            far_dist_path = os.path.join(far_dir, f"{tag}_dist.npy")
            np.save(far_dist_path, distances)
            far_mask_path = os.path.join(far_dir, f"{tag}_mask.npy")
            np.save(far_mask_path, mask)

    with open(os.path.join(args.output_dir, "orders.json"), "w") as f:
        json.dump(all_tag_orders, f)


if __name__ == "__main__":
    main()
