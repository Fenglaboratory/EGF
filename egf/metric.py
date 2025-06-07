from typing import Optional, List, Tuple, Dict
import sys
from tools import smart_rmsd
import numpy as np

blank_metrics = {
    "match_pred": 0.0,
    "match_gt": 0.0,
    "rel_gt": 0.0,
    "rel_match_gt": 0.0,
    "all_match_gt": 0.0,
    "all_rel_match_gt": 0.0,
    "all_rmsd": [[0, 0]],
}


def multihead_metric(
    pred: List[Tuple[str, str]],
    gt: List[Tuple[str, str]],
    match_threshold: float = 2.0,
    precomputed_rmsd: Optional[np.ndarray] = None,
    rmsd_mapping: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    return_all_rmsd: bool = False,
    gt_filters: Optional[List[List[Tuple[int, int]]]] = None,
    ranges: Optional[List[Dict[str, List[Tuple[int, int]]]]] = None,
):
    # pred and gt are lists of pairs: (pdb path, chain)
    # first calculate rmsd between each protein in pred and each protein in gt
    all_metrics = {}

    if precomputed_rmsd is None:
        all_rmsd = []
        for pdb1, chain1 in pred:
            curr_rmsd = []
            for index, (pdb2, chain2) in enumerate(gt):
                if ranges is not None:
                    align_ranges = (
                        ranges[index]["align"] if "align" in ranges[index] else None
                    )
                    rmsd_ranges = (
                        ranges[index]["rmsd"] if "rmsd" in ranges[index] else None
                    )
                else:
                    align_ranges, rmsd_ranges = None, None

                if rmsd_mapping is not None:
                    rmsd = rmsd_mapping[pdb1][pdb2][chain2]
                else:
                    rmsd = smart_rmsd(
                        pdb1,
                        pdb2,
                        chain1=chain1,
                        chain2=chain2,
                        print_stuff=False,
                        filters2=gt_filters[index] if gt_filters is not None else [],
                        align_ranges=align_ranges,
                        rmsd_ranges=rmsd_ranges,
                    )
                curr_rmsd.append(rmsd)
            all_rmsd.append(curr_rmsd)

        all_rmsd = np.array(all_rmsd)
    else:
        all_rmsd = precomputed_rmsd
        pred = [0] * len(precomputed_rmsd)
        gt = [0] * len(precomputed_rmsd[0])

    min_for_pred = np.min(all_rmsd, axis=1)
    min_for_gt = np.min(all_rmsd, axis=0)

    match_pred = np.sum(min_for_pred < match_threshold) / len(pred)
    match_gt = np.sum(min_for_gt < match_threshold) / len(gt)

    all_metrics["match_pred"] = match_pred
    all_metrics["match_gt"] = match_gt

    argmin_for_pred = np.argmin(all_rmsd, axis=1)
    pred_match_mask = min_for_pred < match_threshold

    rel_gt = len(set(argmin_for_pred.tolist())) / len(gt)
    rel_match_gt = len(set(argmin_for_pred[pred_match_mask].tolist())) / len(gt)

    all_metrics["rel_gt"] = rel_gt
    all_metrics["rel_match_gt"] = rel_match_gt

    all_metrics["all_match_gt"] = float(np.any(all_rmsd < match_threshold))
    all_metrics["all_rel_match_gt"] = float(rel_match_gt == 1.0)

    if return_all_rmsd:
        all_metrics["all_rmsd"] = all_rmsd
    return all_metrics


def mean_metrics(all_metrics: List[dict]) -> dict:
    result = {}
    for key in all_metrics[0]:
        if key == "all_rmsd":
            continue
        result[key] = sum([metrics[key] for metrics in all_metrics]) / len(all_metrics)
    return result


def print_metrics(all_metrics: dict):
    print(f"Match Pred: {all_metrics['match_pred']:.3f}")
    print(f"Match GT: {all_metrics['match_gt']:.3f}")
    print(f"Rel GT: {all_metrics['rel_gt']:.3f}")
    print(f"All Match GT: {all_metrics['all_match_gt']:.3f}")
    print(f"Rel Match GT: {all_metrics['rel_match_gt']:.3f}")
    print(f"All Rel Match GT: {all_metrics['all_rel_match_gt']:.3f}")
