from typing import Optional, List, Tuple

root_dir = __file__.split("tools")[0]
import sys

sys.path.append(root_dir)
from tools.align import align
from tools.py_qcprot.py_qcprot import rmsd

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

tto = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}


def extract_ca_from_cif(cif_file, target_chain_id):
    assert cif_file.endswith(".cif")
    dico = MMCIF2Dict(cif_file)
    # dico = {key: np.array(value) for key, value in dico.items()}

    entity_seqs = dico["_entity_poly.pdbx_seq_one_letter_code_can"]
    entity_chains = dico["_entity_poly.pdbx_strand_id"]
    chain_seq_mapping = {}
    assert len(entity_seqs) == len(entity_chains)
    for chain, seq in zip(entity_chains, entity_seqs):
        for split_chain in chain.split(","):
            chain_seq_mapping[split_chain] = seq

    target_sequence = chain_seq_mapping[target_chain_id]

    coords = np.stack(
        [
            dico["_atom_site.Cartn_x"],
            dico["_atom_site.Cartn_y"],
            dico["_atom_site.Cartn_z"],
        ],
        axis=1,
    ).astype(np.float32)
    pdb_groups = np.array(dico["_atom_site.group_PDB"])
    chains = np.array(dico["_atom_site.auth_asym_id"])
    label_seq_ids = np.array(dico["_atom_site.label_seq_id"])
    atom_types = np.array(dico["_atom_site.label_atom_id"])
    model_nums = np.array(dico["_atom_site.pdbx_PDB_model_num"]).astype(np.int32)

    mask = (
        (chains == target_chain_id)
        & (atom_types == "CA")
        & (model_nums == 1)
        & (pdb_groups == "ATOM")
    )
    coords, label_seq_ids = (
        coords[mask].tolist(),
        label_seq_ids[mask].astype(np.int32) - 1,
    )

    return {label_seq_ids[i]: coords[i] for i in range(len(coords))}, str(
        target_sequence
    ).replace("\n", "")


def get_sequence(cif_file, chain):
    protein, sequence = extract_ca_from_cif(cif_file, chain)
    return "".join([sequence[x] for x in sorted(list(protein.keys()))])


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


def align_coords(P, Q, R=None, return_R=False):
    P_, Q_ = P, Q
    p = P_ - P_.mean(0, keepdims=True)
    q = Q_ - Q_.mean(0, keepdims=True)
    if R is None:
        R = kabsch(p, q)
    if return_R:
        return ((P - P_.mean(0, keepdims=True)) @ R) + Q_.mean(0, keepdims=True), R
    else:
        return ((P - P_.mean(0, keepdims=True)) @ R) + Q_.mean(0, keepdims=True)


def rmsd_coords(P, Q):
    diff = P - Q
    return np.sqrt((diff * diff).sum() / P.shape[0])


def fast_smart_rmsd(
    cif1: str,
    cif2: str,
    chain1: str = "A",
    chain2: str = "A",
    protein1=None,
    protein2=None,
    full_sequence1=None,
    full_sequence2=None,
    aligned1=None,
    aligned2=None,
    filters1: List[Tuple[int, int]] = [],
    filters2: List[Tuple[int, int]] = [],
    align_ranges: Optional[List[Tuple[int, int]]] = None,
    rmsd_ranges: Optional[List[Tuple[int, int]]] = None,
):
    if protein1 is None or full_sequence1 is None:
        protein1, full_sequence1 = extract_ca_from_cif(cif1, chain1)
    if protein2 is None or full_sequence2 is None:
        protein2, full_sequence2 = extract_ca_from_cif(cif2, chain2)

    all_exclude_residues = []
    for filter1 in filters1:
        for i in range(filter1[0], filter1[1]):
            all_exclude_residues.append(i)
    all_exclude_residues = set(all_exclude_residues)
    protein1 = {k: v for k, v in protein1.items() if k not in all_exclude_residues}

    all_exclude_residues = []
    for filter2 in filters2:
        for i in range(filter2[0], filter2[1]):
            all_exclude_residues.append(i)
    all_exclude_residues = set(all_exclude_residues)
    protein2 = {k: v for k, v in protein2.items() if k not in all_exclude_residues}

    if len(protein1) < 2000 and len(protein2) < 2000:
        if aligned1 is None or aligned2 is None:
            aligned1, aligned2 = align(full_sequence1, full_sequence2)
        final_aligned1, final_aligned2 = [], []
        for i in range(len(aligned1)):
            if aligned1[i] in protein1 and aligned2[i] in protein2:
                final_aligned1.append(aligned1[i])
                final_aligned2.append(aligned2[i])
        sc1 = [protein1[x] for x in final_aligned1]
        sc2 = [protein2[x] for x in final_aligned2]
        if len(sc1) <= 1:
            return None
        if align_ranges is None and rmsd_ranges is None:
            return rmsd(sc1, sc2)
        else:
            align_ranges = align_ranges if align_ranges is not None else [[0, len(sc1)]]
            rmsd_ranges = rmsd_ranges if rmsd_ranges is not None else [[0, len(sc1)]]

            sc1, sc2 = np.array(sc1), np.array(sc2)
            align_sc1, align_sc2 = [], []
            for r in align_ranges:
                align_sc1.append(sc1[r[0] : r[1]])
                align_sc2.append(sc2[r[0] : r[1]])
            align_sc1 = np.concatenate(align_sc1, axis=0)
            align_sc2 = np.concatenate(align_sc2, axis=0)

            _, R = align_coords(align_sc1, align_sc2, return_R=True)
            sc1, sc2 = align_coords(sc1, sc2, R), sc2

            rmsd_sc1, rmsd_sc2 = [], []
            for r in rmsd_ranges:
                rmsd_sc1.append(sc1[r[0] : r[1]])
                rmsd_sc2.append(sc2[r[0] : r[1]])
            rmsd_sc1 = np.concatenate(rmsd_sc1, axis=0)
            rmsd_sc2 = np.concatenate(rmsd_sc2, axis=0)

            new_rmsd_result = rmsd_coords(rmsd_sc1, rmsd_sc2)
            return new_rmsd_result
    else:
        return None


def smart_rmsd(
    cif1: str,
    cif2: str,
    chain1: str = "A",
    chain2: str = "A",
    print_stuff: bool = True,
    filters1: List[Tuple[int, int]] = [],
    filters2: List[Tuple[int, int]] = [],
    align_ranges: Optional[List[Tuple[int, int]]] = None,
    rmsd_ranges: Optional[List[Tuple[int, int]]] = None,
):
    protein1, full_sequence1 = extract_ca_from_cif(cif1, chain1)
    protein2, full_sequence2 = extract_ca_from_cif(cif2, chain2)

    all_exclude_residues = []
    for filter1 in filters1:
        for i in range(filter1[0], filter1[1]):
            all_exclude_residues.append(i)
    all_exclude_residues = set(all_exclude_residues)
    protein1 = {k: v for k, v in protein1.items() if k not in all_exclude_residues}

    all_exclude_residues = []
    for filter2 in filters2:
        for i in range(filter2[0], filter2[1]):
            all_exclude_residues.append(i)
    all_exclude_residues = set(all_exclude_residues)
    protein2 = {k: v for k, v in protein2.items() if k not in all_exclude_residues}

    if print_stuff:
        print(cif1, cif2, chain1, chain2)
        print(f"Protein 1 {len(protein1)}, Protein 2 {len(protein2)}", flush=True)
    if len(protein1) < 2000 and len(protein2) < 2000:
        aligned1, aligned2 = align(full_sequence1, full_sequence2)
        final_aligned1, final_aligned2 = [], []
        for i in range(len(aligned1)):
            if aligned1[i] in protein1 and aligned2[i] in protein2:
                final_aligned1.append(aligned1[i])
                final_aligned2.append(aligned2[i])

        sc1 = [protein1[x] for x in final_aligned1]
        sc2 = [protein2[x] for x in final_aligned2]
        if len(sc1) <= 1:
            if print_stuff:
                print("Skipped due to no alignment.")
            return None
        if print_stuff:
            print(f"Aligned Residues: {len(sc1)}")
        if align_ranges is None and rmsd_ranges is None:
            return rmsd(sc1, sc2)
        else:
            align_ranges = align_ranges if align_ranges is not None else [[0, len(sc1)]]
            rmsd_ranges = rmsd_ranges if rmsd_ranges is not None else [[0, len(sc1)]]

            sc1, sc2 = np.array(sc1), np.array(sc2)
            align_sc1, align_sc2 = [], []
            for r in align_ranges:
                align_sc1.append(sc1[r[0] : r[1]])
                align_sc2.append(sc2[r[0] : r[1]])
            align_sc1 = np.concatenate(align_sc1, axis=0)
            align_sc2 = np.concatenate(align_sc2, axis=0)

            _, R = align_coords(align_sc1, align_sc2, return_R=True)
            sc1, sc2 = align_coords(sc1, sc2, R), sc2

            rmsd_sc1, rmsd_sc2 = [], []
            for r in rmsd_ranges:
                rmsd_sc1.append(sc1[r[0] : r[1]])
                rmsd_sc2.append(sc2[r[0] : r[1]])
            rmsd_sc1 = np.concatenate(rmsd_sc1, axis=0)
            rmsd_sc2 = np.concatenate(rmsd_sc2, axis=0)

            new_rmsd_result = rmsd_coords(rmsd_sc1, rmsd_sc2)
            return new_rmsd_result
    else:
        if print_stuff:
            print("Skipped due to length.")


from colorama import Fore


def show_seqs(seq1, seq2, align_fn=align):
    a1, a2 = align_fn(seq1, seq2)
    mapped_indices = [max(a1[0], a2[0]) + 1]
    for i in range(1, len(a1)):
        mapped_indices.append(
            mapped_indices[-1] + max(a1[i] - a1[i - 1], a2[i] - a2[i - 1])
        )
    mapped_indices.append(
        mapped_indices[-1] + max(len(seq1) - a1[-1], len(seq2) - a2[-1])
    )
    result1 = [" "] * (mapped_indices[-1] + 1)
    result2 = [" "] * (mapped_indices[-1] + 1)
    for i in range(len(a1)):
        result1[mapped_indices[i]] = Fore.GREEN + seq1[a1[i]] + Fore.RESET
        result2[mapped_indices[i]] = Fore.GREEN + seq2[a2[i]] + Fore.RESET
    for i in range(len(mapped_indices)):
        for j in range(
            0 if i == 0 else a1[i - 1] + 1, a1[i] if i != len(a1) else len(seq1)
        ):
            diff = j - (0 if i == 0 else a1[i - 1] + 1)
            result1[(0 if i == 0 else mapped_indices[i - 1]) + 1 + diff] = (
                Fore.RED + seq1[j] + Fore.RESET
            )
        for j in range(
            0 if i == 0 else a2[i - 1] + 1, a2[i] if i != len(a2) else len(seq2)
        ):
            diff = j - (0 if i == 0 else a2[i - 1] + 1)
            result2[(0 if i == 0 else mapped_indices[i - 1]) + 1 + diff] = (
                Fore.RED + seq2[j] + Fore.RESET
            )

    print("".join(result1))
    print("".join(result2))


from openfold.np import protein


def convert_pdb_to_cif(input_pdb_path: str, output_cif_path: str, pdb_chain: str = "A"):
    assert input_pdb_path.endswith(".pdb")
    assert output_cif_path.endswith(".cif")
    with open(input_pdb_path, "r") as f:
        pdb_text = f.read()
    prot = protein.from_pdb_string(pdb_text, pdb_chain)
    cif_text = protein.to_modelcif(prot)
    with open(output_cif_path, "w") as f:
        f.write(cif_text)
