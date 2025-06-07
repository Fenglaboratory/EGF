from typing import Optional, List, Tuple
from sklearn.decomposition import PCA
import numpy as np
from tools import extract_ca_from_cif


def kabsch(a, b, weights=None, return_v=False):
    a = np.asarray(a)
    b = np.asarray(b)
    if weights is None:
        weights = np.ones(len(b))
    else:
        weights = np.asarray(weights)
    B = np.einsum("ji,jk->ik", weights[:, None] * a, b)
    u, s, vh = np.linalg.svd(B)
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] = -u[:, -1]
    if return_v:
        return u
    else:
        return u @ vh


Ls = None


def align(P, Q):
    if Ls is None or len(Ls) == 1:
        P_, Q_ = P, Q
    else:
        # align relative to first chain
        P_, Q_ = P[: Ls[0]], Q[: Ls[0]]
    p = P_ - P_.mean(0, keepdims=True)
    q = Q_ - Q_.mean(0, keepdims=True)
    return ((P - P_.mean(0, keepdims=True)) @ kabsch(p, q)) + Q_.mean(0, keepdims=True)


def get_aligned_coords(input_structure_paths, input_chains):
    coords = []
    for input_structure_path, input_chain in zip(input_structure_paths, input_chains):
        curr_coords, curr_seq = extract_ca_from_cif(input_structure_path, input_chain)
        coords_list = [curr_coords[i] for i in range(len(curr_seq))]
        coords.append(coords_list)
    coords = [np.array(coord) for coord in coords]
    coord_0 = coords[0].copy()
    aligned_coords = [align(coord, coord_0) for coord in coords]
    aligned_coords = np.stack(aligned_coords, axis=0)
    return aligned_coords


def pca_reduction(
    pred_paths: List[str], reduce_num: Optional[int] = 1
) -> Tuple[List[str], np.ndarray]:
    pca = PCA(n_components=1)
    N = len(pred_paths)
    coords = get_aligned_coords(pred_paths, ["A"] * len(pred_paths))
    reduced = pca.fit_transform(coords.reshape([N, -1]))

    sorted_indices = np.argsort(reduced[:, 0], axis=0)
    extreme_indices = np.concatenate(
        [sorted_indices[:reduce_num], sorted_indices[-reduce_num:]], axis=0
    )
    all_paths = [pred_paths[x] for x in extreme_indices]

    return all_paths, reduced[extreme_indices]
