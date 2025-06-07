# Adapted from https://github.com/aqlaboratory/openfold/blob/main/run_pretrained_openfold.py

import argparse
import logging
import math
import numpy as np
import os
import ast
import wandb
import sys
import datetime
import pickle
import random
import time
import torch
import glob
import json
from egf.utils import random_seed
from egf.plot import plot_rmsd_path
from egf.download import download_structure
from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
    run_model,
    prep_output,
)  # , relax_protein

from omegaconf import OmegaConf

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)

TRACING_INTERVAL = 50


def precompute_alignments(tags, seqs, data_dirs, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(
            data_dirs["output_dir"], f"tmp_{os.getpid()}.fasta"
        )
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(data_dirs["alignment_dir"], tag)
        if data_dirs["alignment_dir"] is None and not os.path.isdir(
            local_alignment_dir
        ):
            logger.info(f"Generating alignments for {tag}...")

            os.makedirs(local_alignment_dir)

            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                no_cpus=args.cpus,
            )
            alignment_runner.run(tmp_fasta_path, local_alignment_dir)
        else:
            logger.info(
                f"Using precomputed alignments for {tag} at {data_dirs['alignment_dir']}..."
            )

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def generate_feature_dict(
    tags,
    seqs,
    data_dirs,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(data_dirs["output_dir"], f"tmp_{os.getpid()}.fasta")
    if len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(data_dirs["alignment_dir"], tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=(args.config_preset[:3] == "seq"),
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write("\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path,
            super_alignment_dir=data_dirs["alignment_dir"],
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


class AlphaFold:
    def __init__(self, args, guide_config, full_config=None):
        self.guide_config = guide_config
        if (
            hasattr(self.guide_config, "tag_cluster_mapping_path")
            and self.guide_config.tag_cluster_mapping_path is not None
        ):
            with open(self.guide_config.tag_cluster_mapping_path, "r") as f:
                self.tag_cluster_mapping = json.load(f)
        else:
            self.tag_cluster_mapping = None
        self.full_config = full_config
        self.args = args
        self.config = model_config(
            self.args.config_preset,
            long_sequence_inference=self.args.long_sequence_inference,
        )
        self.config.globals.kalign_binary_path = self.args.kalign_binary_path
        self.config.globals.use_mek = self.args.use_mek
        self.config.data.common.max_recycling_iters = (
            self.guide_config.max_recycling_iters - 1
        )
        self.model_generator = list(
            load_models_from_command_line(
                self.config,
                self.args.model_device,
                self.args.openfold_checkpoint_path,
                self.args.jax_param_path,
                output_dir="",
                guide_config=self.guide_config,
            )
        )
        assert self.args.cif_output

    def preprocess(self, tag, seq, data_dirs):
        self.feature_dicts = {}
        template_featurizer = (
            templates.TemplateHitFeaturizer(
                mmcif_dir=data_dirs["template_dir"],
                max_template_date=self.args.max_template_date,
                max_hits=self.config.data.predict.max_templates,
                kalign_binary_path=self.args.kalign_binary_path,
                release_dates_path=self.args.release_dates_path,
                obsolete_pdbs_path=self.args.obsolete_pdbs_path,
            )
            if len(glob.glob(data_dirs["template_dir"] + "/*")) > 0
            else None
        )

        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        precompute_alignments([tag], [seq], data_dirs, self.args)

        feature_dict = self.feature_dicts.get(tag, None)
        if feature_dict is None:
            feature_dict = generate_feature_dict(
                [tag],
                [seq],
                data_dirs,
                data_processor,
                self.args,
            )

            if self.args.trace_model:
                n = feature_dict["aatype"].shape[-2]
                rounded_seqlen = round_up_seqlen(n)
                feature_dict = pad_feature_dict_seq(
                    feature_dict,
                    rounded_seqlen,
                )

            self.feature_dicts[tag] = feature_dict

        processed_feature_dict = self.feature_processor.process_features(
            feature_dict,
            mode="predict",
        )

        processed_feature_dict = {
            k: torch.as_tensor(v, device=self.args.model_device)
            for k, v in processed_feature_dict.items()
        }

        return feature_dict, processed_feature_dict

    def postprocess(
        self,
        feature_dict,
        processed_feature_dict,
        out,
        unrelaxed_output_path,
        output_directory,
        tag,
    ):
        processed_feature_dict = tensor_tree_map(
            lambda x: np.array(x[..., -1].cpu()), processed_feature_dict
        )
        out = tensor_tree_map(lambda x: np.array(x.cpu().type(torch.float32)), out)

        unrelaxed_protein = prep_output(
            out,
            processed_feature_dict,
            feature_dict,
            self.feature_processor,
            self.args.config_preset,
            self.args.multimer_ri_gap,
            self.args.subtract_plddt,
        )

        with open(unrelaxed_output_path, "w") as fp:
            fp.write(protein.to_pdb(unrelaxed_protein))
        if self.args.cif_output:
            cif_output_path = unrelaxed_output_path.replace(".pdb", ".cif")
            with open(cif_output_path, "w") as fp:
                fp.write(protein.to_modelcif(unrelaxed_protein))

        if (
            not self.args.skip_relaxation
            and output_directory is not None
            and tag is not None
        ):
            # Relax the prediction.
            logger.info(f"Running relaxation on {unrelaxed_output_path}...")
            relax_protein(
                self.config,
                self.args.model_device,
                unrelaxed_protein,
                output_directory,
                tag,
                self.args.cif_output,
            )

    def run(self, input_dict, data_dirs):
        gt_structures = {"p1": self.full_config.base.p1, "p2": self.full_config.base.p2}
        # Create the output directory
        os.makedirs(data_dirs["output_dir"], exist_ok=True)
        output_dir_base = data_dirs["output_dir"]
        if not os.path.exists(output_dir_base):
            os.makedirs(output_dir_base)

        tag_list = []
        seq_list = []
        for key, value in input_dict.items():
            tag_list.append(key)
            seq_list.append(value)

        seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
        sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
        output_paths = {}
        intermediate_paths = {}
        for model, _ in self.model_generator:
            config_output_path = os.path.join(output_dir_base, "config.yaml")
            with open(config_output_path, "w") as f:
                OmegaConf.save(model.guide_config, f)

            model.device_group = (
                self.guide_config.device_group
                if self.guide_config.device_group is not None
                else None
            )
            for tag, seq in sorted_targets:
                random_seed(self.args.data_random_seed)
                unrelaxed_output_path = os.path.join(
                    output_dir_base,
                    f"{tag}.cif" if self.args.cif_output else f"{tag}.pdb",
                )
                if (
                    os.path.exists(unrelaxed_output_path)
                    and self.guide_config.skip_existing
                ):
                    continue
                if tag not in self.tag_cluster_mapping:
                    continue
                feature_dict, processed_feature_dict = self.preprocess(
                    tag, seq, data_dirs
                )

                curr_intermediate_paths = []
                out, intermediate_outputs, intermediate_metrics = run_model(
                    model, processed_feature_dict, tag, output_dir_base
                )
                metrics_save_path = os.path.join(
                    self.guide_config.info_dir, f"{tag}_metrics.json"
                )
                with open(metrics_save_path, "w") as f:
                    json.dump(intermediate_metrics, f)

                for intermediate_index, intermediate_structure in enumerate(
                    intermediate_outputs
                ):
                    os.makedirs(self.guide_config.info_dir, exist_ok=True)

                    intermediate_path = os.path.join(
                        self.guide_config.info_dir,
                        f"{tag}_int_{intermediate_index}.pdb",
                    )
                    curr_intermediate_paths.append(
                        intermediate_path.replace(".pdb", ".cif")
                    )
                    self.postprocess(
                        feature_dict,
                        processed_feature_dict,
                        intermediate_structure,
                        intermediate_path,
                        None,
                        None,
                    )

                metadata_output_path = os.path.join(
                    self.guide_config.info_dir, f"{tag}_metadata.txt"
                )
                with open(metadata_output_path, "w") as f:
                    for intermediate_index, intermediate_structure in enumerate(
                        intermediate_outputs
                    ):
                        f.write(
                            f"{intermediate_index}\t{intermediate_structure['true_plddt']:.3f}\t{intermediate_structure['tmscore']:.3f}\n"
                        )
                intermediate_paths[tag] = curr_intermediate_paths

                p1, p2 = (
                    self.tag_cluster_mapping[tag][0],
                    self.tag_cluster_mapping[tag][1],
                )
                if p1 is not None:
                    download_structure(p1, data_dirs["structure_dir"])
                if p2 is not None:
                    download_structure(p2, data_dirs["structure_dir"])

                if (
                    self.guide_config.plot_rmsd_path
                    and p1 is not None
                    and p2 is not None
                ):
                    plot_rmsd_path(
                        curr_intermediate_paths,
                        output_path=os.path.join(
                            self.guide_config.info_dir, f"{tag}_rmsd_path.png"
                        ),
                        structure_dir=data_dirs["structure_dir"],
                        p1=self.tag_cluster_mapping[tag][0],
                        p2=self.tag_cluster_mapping[tag][1],
                    )

                if tag not in output_paths:
                    output_paths[tag] = []
                output_paths[tag].append(unrelaxed_output_path)
                self.postprocess(
                    feature_dict,
                    processed_feature_dict,
                    out,
                    unrelaxed_output_path,
                    output_dir_base,
                    tag,
                )

                # Toss out the recycling dimensions --- we don't need them anymore
                logger.info(f"Output written to {unrelaxed_output_path}...")
                if self.args.save_outputs:
                    output_dict_path = os.path.join(
                        output_dir_base, f"{tag}_output_dict.pkl"
                    )
                    with open(output_dict_path, "wb") as fp:
                        pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    logger.info(f"Model output written to {output_dict_path}...")
        return output_paths


def get_base_model(config):
    return AlphaFold(config.base, config.guide_config, full_config=config)
