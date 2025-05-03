import os
import sys
import json
import argparse
import tqdm

from egf.download import download_msa, download_structure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--tag_path", type=str, required=True)
    parser.add_argument("--sequence_dict_path", type=str, required=True)

    # sequence_dir, alignment_dir, structure_dir, and temp_dir, which are not required
    parser.add_argument("--sequence_dir", type=str, default=None)
    parser.add_argument("--alignment_dir", type=str, default=None)
    parser.add_argument("--structure_dir", type=str, default=None)
    parser.add_argument("--temp_dir", type=str, default=None)

    # Options msa, structure, and sequence which are all store_true
    parser.add_argument("--msa", action="store_true")
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--sequence", action="store_true")

    # Option all which is store_true
    parser.add_argument("--all", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.tag_path, "r") as fp:
        tags = json.load(fp)

    with open(args.sequence_dict_path, "r") as fp:
        sequence_dict = json.load(fp)

    sequence_dir = (
        args.sequence_dir
        if args.sequence_dir is not None
        else os.path.join(args.root_dir, "sequences")
    )
    alignment_dir = (
        args.alignment_dir
        if args.alignment_dir is not None
        else os.path.join(args.root_dir, "alignments")
    )
    structure_dir = (
        args.structure_dir
        if args.structure_dir is not None
        else os.path.join(args.root_dir, "structures")
    )
    temp_dir = (
        args.temp_dir
        if args.temp_dir is not None
        else os.path.join(args.root_dir, "temp")
    )

    if args.all:
        args.msa = True
        args.structure = True
        args.sequence = True

    if args.msa:
        for tag in tqdm.tqdm(tags):
            download_msa(tag, alignment_dir, sequence_dict, temp_dir)

    if args.structure:
        for tag in tqdm.tqdm(tags):
            download_structure(tag, structure_dir)

    if args.sequence:
        os.makedirs(sequence_dir, exist_ok=True)
        for tag in tqdm.tqdm(tags):
            with open(os.path.join(sequence_dir, f"{tag}.fasta"), "w") as fp:
                fp.write(f">{tag}\n{sequence_dict[tag]}")


if __name__ == "__main__":
    main()
