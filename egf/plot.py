import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tools import smart_rmsd


def plot_rmsd_path(pred_paths, output_path, structure_dir, p1, p2):
    p1_rmsds = []
    p2_rmsds = []
    for path in pred_paths:
        p1_path = os.path.join(structure_dir, f"{p1[:4]}.cif")
        p2_path = os.path.join(structure_dir, f"{p2[:4]}.cif")
        p1_rmsd = smart_rmsd(path, p1_path, chain1="A", chain2=p1[5:])
        p2_rmsd = smart_rmsd(path, p2_path, chain1="A", chain2=p2[5:])
        p1_rmsds.append(p1_rmsd)
        p2_rmsds.append(p2_rmsd)

    plt.scatter(p1_rmsds, p2_rmsds)
    for n, (p1_rmsd, p2_rmsd) in enumerate(zip(p1_rmsds, p2_rmsds)):
        plt.annotate(str(n), (p1_rmsd, p2_rmsd))
    for i in range(len(pred_paths) - 1):
        plt.arrow(
            p1_rmsds[i],
            p2_rmsds[i],
            p1_rmsds[i + 1] - p1_rmsds[i],
            p2_rmsds[i + 1] - p2_rmsds[i],
            head_width=0.03,
            color="blue",
            length_includes_head=True,
        )
    plt.title("RMSD Path")
    plt.xlabel(f"{p1} RMSD")
    plt.ylabel(f"{p2} RMSD")
    plt.savefig(output_path, dpi=300)
    plt.clf()
