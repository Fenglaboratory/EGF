from egf.metric import multihead_metric, mean_metrics, print_metrics, blank_metrics
from egf.postprocess import pca_reduction
import os
import tqdm
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_file_pattern",
        type=str,
        required=True,
        help="Pattern for prediction files",
    )
    parser.add_argument(
        "--tag_cluster_mapping",
        type=str,
        required=True,
        help="Path to tag cluster mapping file",
    )
    parser.add_argument(
        "--pca_reduction", action="store_true", help="Whether to perform PCA reduction"
    )
    parser.add_argument(
        "--reduce_num", type=int, default=1, help="Number of predictions to reduce"
    )
    parser.add_argument(
        "--match_threshold", type=float, default=2.0, help="Threshold for matching"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save output"
    )
    parser.add_argument(
        "--structure_dir", type=str, required=True, help="Structure folder"
    )
    parser.add_argument(
        "--precomputed_rmsd_path",
        type=str,
        default=None,
        help="Path to precomputed rmsd npz file",
    )
    parser.add_argument(
        "--rmsd_mapping_path",
        type=str,
        default=None,
        help="Path to precomputed rmsd json mapping",
    )
    parser.add_argument(
        "--gt_filter_mapping_path",
        type=str,
        default=None,
        help="Path to filters on protein structures",
    )
    parser.add_argument(
        "--range_mapping_path",
        type=str,
        default=None,
        help="Path to alignment and RMSD ranges",
    )

    parser.add_argument(
        "--scatter_rmsds", action="store_true", help="Whether to scatter RMSDs"
    )
    parser.add_argument(
        "--pca_coloring", action="store_true", help="Whether to color by PCA"
    )
    parser.add_argument(
        "--output_plot_pattern", type=str, help="Pattern for output plot"
    )

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.tag_cluster_mapping, "r") as f:
        tag_cluster_mapping = json.load(f)

    if args.rmsd_mapping_path is not None:
        with open(args.rmsd_mapping_path, "r") as f:
            rmsd_mapping = json.load(f)
    else:
        rmsd_mapping = None

    if args.gt_filter_mapping_path is not None:
        with open(args.gt_filter_mapping_path, "r") as f:
            gt_filter_mapping = json.load(f)
    else:
        gt_filter_mapping = {}

    if args.range_mapping_path is not None:
        with open(args.range_mapping_path, "r") as f:
            range_mapping = json.load(f)
    else:
        range_mapping = {}

    if args.precomputed_rmsd_path is not None:
        precomputed_rmsds = np.load(args.precomputed_rmsd_path)

    all_metrics = {}
    for tag, cluster in tag_cluster_mapping.items():
        print(tag)
        pred_path = glob.glob(args.pred_file_pattern.format(tag=tag))
        pred_path.sort()

        if len(pred_path) == 0:
            print(blank_metrics["all_rmsd"])
            print_metrics(blank_metrics)
            all_metrics["tag"] = blank_metrics
            continue

        if args.pca_reduction:
            pred_path, pca_values = pca_reduction(pred_path, args.reduce_num)
        gt_paths = [
            (os.path.join(args.structure_dir, f"{pdb_id[:4]}.cif"), pdb_id[5:])
            for pdb_id in cluster
        ]
        gt_filters = [
            gt_filter_mapping[pdb_id] if pdb_id in gt_filter_mapping else []
            for pdb_id in cluster
        ]
        ranges = [
            range_mapping[pdb_id] if pdb_id in range_mapping else {}
            for pdb_id in cluster
        ]

        curr_metrics = multihead_metric(
            [(path, "A") for path in pred_path],
            gt_paths,
            match_threshold=args.match_threshold,
            return_all_rmsd=True,
            precomputed_rmsd=precomputed_rmsds[tag]
            if args.precomputed_rmsd_path is not None and tag in precomputed_rmsds
            else None,
            rmsd_mapping=rmsd_mapping,
            gt_filters=gt_filters,
            ranges=ranges,
        )
        print_metrics(curr_metrics)
        if "all_rmsd" in curr_metrics:
            curr_metrics["all_rmsd"] = curr_metrics["all_rmsd"].tolist()
        all_metrics[tag] = curr_metrics

        if args.scatter_rmsds:
            rmsds = np.array(curr_metrics["all_rmsd"])

            if args.pca_coloring:
                normalized_pca_values = (pca_values - pca_values.min()) / (
                    pca_values.max() - pca_values.min()
                )
                plt.scatter(rmsds[:, 0], rmsds[:, 1], c=normalized_pca_values)
            else:
                plt.scatter(rmsds[:, 0], rmsds[:, 1])
            plt.xlabel(f"RMSD to {cluster[0]}")
            plt.ylabel(f"RMSD to {cluster[1]}")
            plt.title(f"RMSD of predictions for {tag}")
            plt.savefig(args.output_plot_pattern.format(tag=tag), dpi=600)
            plt.clf()

    final_metrics = mean_metrics(list(all_metrics.values()))
    print("Final metrics")
    print_metrics(final_metrics)

    print(f"Saving metrics to {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(all_metrics, f)


if __name__ == "__main__":
    main()
