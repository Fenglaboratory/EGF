import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from tools import smart_rmsd
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag_cluster_mapping", type=str, required=True)
    parser.add_argument("--orders_path", type=str, required=True)
    parser.add_argument("--output_reduction_path", type=str, required=True)
    parser.add_argument("--output_rmsd_path", type=str, required=True)
    parser.add_argument(
        "--pred_pattern", type=str, required=True
    )  # format: path with {tag} and {step} inside.
    parser.add_argument("--structure_dir", type=str, required=True)
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

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.tag_cluster_mapping, "r") as f:
        tcm = json.load(f)
    with open(args.orders_path, "r") as f:
        orders = json.load(f)

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

    rmsds = {tag: [] for tag in tcm}
    for i in range(11):
        for tag, cluster in tcm.items():
            cluster_tag = orders[tag][1]
            if cluster_tag in range_mapping:
                align_ranges = (
                    range_mapping[cluster_tag]["align"]
                    if "align" in range_mapping[cluster_tag]
                    else None
                )
                rmsd_ranges = (
                    range_mapping[cluster_tag]["rmsd"]
                    if "rmsd" in range_mapping[cluster_tag]
                    else None
                )
            else:
                align_ranges, rmsd_ranges = None, None
            rmsd = smart_rmsd(
                os.path.join(args.structure_dir, f"{orders[tag][1][:4]}.cif"),
                args.pred_pattern.format(tag=tag, step=i),
                align_ranges=align_ranges,
                rmsd_ranges=rmsd_ranges,
                filters2=gt_filter_mapping[cluster_tag]
                if cluster_tag in gt_filter_mapping
                else [],
                chain1=orders[tag][1][5:],
                chain2="A",
                print_stuff=False,
            )
            rmsds[tag].append(rmsd)

    data = {key: [rmsds[key][0], rmsds[key][-1]] for key in rmsds}
    data["average"] = [
        np.mean([rmsds[key][0] for key in rmsds]),
        np.mean([rmsds[key][-1] for key in rmsds]),
    ]

    # Prepare the plot
    fig, ax = plt.subplots()

    # Define colors
    before_color = "blue"
    after_color = "red"

    # Loop through the data and plot points and arrows
    for i, (key, values) in enumerate(data.items()):
        before, after = values
        ax.plot(i, before, "o", color=before_color, label="Before" if i == 0 else "")
        ax.plot(i, after, "o", color=after_color, label="After" if i == 0 else "")
        ax.annotate(
            "",
            xy=(i, after),
            xytext=(i, before),
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # Set the x-axis labels to the keys
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.keys(), rotation=90)

    # Add labels and title
    ax.set_ylabel("RMSD (Å)")
    ax.set_xlabel("PDB ID of pair")
    ax.set_title("RMSD to target conformation before and after distogram guidance")

    # Add legend
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(args.output_reduction_path, dpi=600)
    plt.clf()

    data = rmsds

    # Calculate the average for each index
    all_values = np.array(list(data.values()))
    average = np.mean(all_values, axis=0)

    # Plot each line in gray
    x = range(len(next(iter(data.values()))))
    for label, values in data.items():
        plt.plot(x, values, label=label, color="lightgray")
        plt.text(x[0] + 0.05, values[0], label, fontsize=5, verticalalignment="center")

    # Plot the average line in a different color
    plt.plot(x, average, label="average", color="orangered", linewidth=2)
    plt.text(
        x[-3],
        average[-3] + 0.2,
        "average",
        fontsize=9,
        verticalalignment="center",
        color="orangered",
    )
    ax = plt.gca()
    ax.set_xlim([0, 10])
    # Show the plot
    plt.xlabel("Step")
    plt.ylabel("RMSD (Å)")
    plt.yticks(list(range(12)))
    plt.title("RMSD to target alternative conformation for different steps")
    plt.savefig(args.output_rmsd_path, dpi=600)
    plt.clf()


if __name__ == "__main__":
    main()
