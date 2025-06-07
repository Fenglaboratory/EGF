import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse


def entropy(logits):
    logits = logits.reshape([-1, logits.shape[-1]])
    return (-torch.log_softmax(logits, dim=-1) * logits.softmax(dim=-1)).sum(dim=-1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag_cluster_mapping", type=str, required=True)
    parser.add_argument(
        "--distogram_pattern",
        type=str,
        required=True,
        help="Format: path with {tag} inside.",
    )
    parser.add_argument("--entropy_increase_output_path", type=str, required=True)
    parser.add_argument("--entropy_change_output_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.tag_cluster_mapping, "r") as f:
        tcm = json.load(f)

    entropys = {}
    for tag in tcm:
        distograms = torch.load(args.distogram_pattern.format(tag=tag))
        entropy_before = entropy(distograms[0]).mean().to(torch.float32)
        entropy_after = entropy(distograms[1]).mean().to(torch.float32)
        entropys[tag] = [entropy_before, entropy_after]

    data = entropys
    data["average"] = [
        np.mean([entropys[key][0] for key in entropys]),
        np.mean([entropys[key][-1] for key in entropys]),
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
    ax.set_ylabel("Entropy")
    ax.set_xlabel("PDB ID of pair")
    ax.set_title("Entropy of distograms before and after entropy guidance")

    # Add legend
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(args.entropy_increase_output_path, dpi=600)
    plt.clf()

    # https://stackoverflow.com/questions/1340338/back-to-back-histograms-in-matplotlib
    distograms = torch.load(args.distogram_pattern.format(tag="2NRF_A"))
    dataOne = entropy(distograms[0]).to(torch.float32)
    dataTwo = entropy(distograms[1]).to(torch.float32)

    hN = plt.hist(dataTwo, orientation="horizontal", rwidth=0.8, label="After", bins=30)
    hS = plt.hist(
        dataOne, bins=hN[1], orientation="horizontal", rwidth=0.8, label="Before"
    )

    for p in hS[2]:
        p.set_width(-p.get_width())

    xmin = min([min(w.get_width() for w in hS[2]), min([w.get_width() for w in hN[2]])])
    xmin = np.floor(xmin)
    xmax = max([max(w.get_width() for w in hS[2]), max([w.get_width() for w in hN[2]])])
    xmax = np.ceil(xmax)
    range_x = xmax - xmin
    delta = 0.0 * range_x
    plt.xlim([xmin - delta, xmax + delta])
    xt = plt.xticks()
    n = xt[0]
    s = [int(abs(i)) for i in n]
    plt.xticks(n, s)
    plt.legend(loc="best")
    plt.title("Entropy of distograms before and after entropy guidance (2NRF_A)")
    plt.xlabel("Frequency")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(args.entropy_change_output_path, dpi=600)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    main()
