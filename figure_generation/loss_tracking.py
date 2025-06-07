import numpy as np
import itertools
import matplotlib.pyplot as plt
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag_cluster_mapping", type=str, required=True)
    parser.add_argument(
        "--metrics_pattern", type=str, required=True, help="Pattern for metrics files"
    )
    parser.add_argument("--sample_output_path", type=str, required=True)
    parser.add_argument("--mean_output_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.tag_cluster_mapping, "r") as f:
        tcm = json.load(f)

    all_entropy_losses = []
    all_distogram_losses = []
    all_tags = []
    sample_index = None
    for index, tag in enumerate(tcm):
        if tag == "7QPA_A":
            sample_index = index
        metric_path = args.metrics_pattern.format(tag=tag)
        with open(metric_path, "r") as f:
            metrics = json.load(f)
        entropy_losses = [x[0][1] for x in metrics]
        distogram_losses = [x[1][1] for x in metrics]
        entropy_losses = np.array(entropy_losses) - entropy_losses[0]
        distogram_losses = np.array(distogram_losses) - distogram_losses[0]
        all_entropy_losses.append(entropy_losses)
        all_distogram_losses.append(distogram_losses)
        all_tags.append(tag)

    (h1,) = plt.plot(all_entropy_losses[sample_index], color="orange")
    (h2,) = plt.plot(all_distogram_losses[sample_index], color="blue")
    plt.legend([h1, h2], ["Entropy Loss", "Distogram Loss"], loc="best")
    plt.xlabel("Steps")
    plt.ylabel("Normalized loss")
    plt.xticks(list(range(len(all_entropy_losses[0]))))
    plt.title(
        f"Entropy and distogram loss after steps of entropy guidance for {all_tags[sample_index]}"
    )

    plt.savefig(args.sample_output_path, dpi=600)
    plt.clf()

    all_entropy_losses = np.array(all_entropy_losses)
    all_distogram_losses = np.array(all_distogram_losses)

    N, steps = all_entropy_losses.shape[0], all_entropy_losses.shape[1]

    all_x = list(itertools.chain(*[[i for _ in range(N)] for i in range(steps)]))
    plt.scatter(all_x, all_distogram_losses.T.flatten(), color="lightblue")
    plt.scatter(all_x, all_entropy_losses.T.flatten(), color="wheat")

    h1 = plt.errorbar(
        list(range(steps)),
        all_distogram_losses.T.mean(axis=1).flatten(),
        yerr=all_distogram_losses.T.std(axis=1).flatten(),
        color="blue",
        marker="o",
        capsize=3,
    )
    h2 = plt.errorbar(
        list(range(steps)),
        all_entropy_losses.T.mean(axis=1).flatten(),
        yerr=all_entropy_losses.T.std(axis=1).flatten(),
        color="orange",
        marker="o",
        capsize=3,
    )

    plt.xlabel("Steps")
    plt.ylabel("Normalized loss")
    plt.xticks(list(range(steps)))
    plt.legend([h1, h2], ["Entropy Loss", "Distogram Loss"], loc="best")
    plt.title(f"Entropy and distogram loss after steps of entropy guidance")

    plt.savefig(args.mean_output_path, dpi=600)
    plt.clf()


if __name__ == "__main__":
    main()
