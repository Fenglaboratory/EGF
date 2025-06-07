# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, List, Tuple, Dict
from functools import partial
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.data import data_transforms_multimer
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    dgram_from_positions,
    atom14_to_atom37,
)
from openfold.utils.tensor_utils import masked_mean
from openfold.model.embedders import (
    InputEmbedder,
    InputEmbedderMultimer,
    RecyclingEmbedder,
    TemplateEmbedder,
    TemplateEmbedderMultimer,
    ExtraMSAEmbedder,
    PreembeddingEmbedder,
)
from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.model.heads import AuxiliaryHeads
from openfold.model.structure_module import StructureModule
from openfold.model.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
    embed_templates_average,
    embed_templates_offload,
)
import openfold.np.residue_constants as residue_constants
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
from openfold.utils.loss import (
    compute_plddt,
)
from openfold.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)
from openfold.utils.loss import compute_tm
from openfold.utils.tensor_utils import dists_to_distogram

import os
import time
import copy
from tabulate import tabulate
import numpy as np

str_to_dtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


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


def align(P, Q, R=None, return_R=False):
    P_, Q_ = P, Q
    p = P_ - P_.mean(0, keepdims=True)
    q = Q_ - Q_.mean(0, keepdims=True)
    if R is None:
        R = kabsch(p, q)
    if return_R:
        return ((P - P_.mean(0, keepdims=True)) @ R) + Q_.mean(0, keepdims=True), R
    else:
        return ((P - P_.mean(0, keepdims=True)) @ R) + Q_.mean(0, keepdims=True)


def rmsd(P, Q):
    diff = P - Q
    return np.sqrt((diff * diff).sum() / P.shape[0])


def full_rmsd(P, Q):
    return rmsd(P, align(Q, P))


def entropy(x, sum_dim=-1):
    return (x * torch.log(x + 1e-4)).sum(dim=sum_dim)


class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config, guide_config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold, self).__init__()

        self.guide_config = guide_config
        self.full_config = config
        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template
        self.extra_msa_config = self.config.extra_msa
        self.seqemb_mode = config.globals.seqemb_mode_enabled

        # Main trunk + structure module
        if self.globals.is_multimer:
            self.input_embedder = InputEmbedderMultimer(**self.config["input_embedder"])
        elif self.seqemb_mode:
            # If using seqemb mode, embed the sequence embeddings passed
            # to the model ("preembeddings") instead of embedding the sequence
            self.input_embedder = PreembeddingEmbedder(
                **self.config["preembedding_embedder"],
            )
        else:
            self.input_embedder = InputEmbedder(
                **self.config["input_embedder"],
            )

        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        if self.template_config.enabled:
            if self.globals.is_multimer:
                self.template_embedder = TemplateEmbedderMultimer(
                    self.template_config,
                )
            else:
                self.template_embedder = TemplateEmbedder(
                    self.template_config,
                )

        if self.extra_msa_config.enabled:
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_config["extra_msa_stack"],
            )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        self.structure_module = StructureModule(
            is_multimer=self.globals.is_multimer,
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

        self.orig_blocks_per_ckpt = self.globals.blocks_per_ckpt
        self.evoformer.eval()
        self.structure_module.eval()
        for param in self.evoformer.parameters():
            param.requires_grad = False
        for param in self.aux_heads.distogram.parameters():
            param.requires_grad = False
        for param in self.structure_module.parameters():
            param.requires_grad = False

    def reverse_opt_settings(self):
        self.evoformer = self.evoformer.to(torch.float32)
        self.structure_module = self.structure_module.to(torch.float32)
        self.aux_heads = self.aux_heads.to(torch.float32)
        self.recycling_embedder = self.recycling_embedder.to(torch.float32)
        self.extra_msa_stack = self.extra_msa_stack.to(torch.float32)

        self.globals.offload_inference = False
        self.globals.use_deepspeed_evo_attention = False
        self.config.template.offload_inference = False
        self.config.template.template_pair_stack.tune_chunk_size = True
        self.config.extra_msa.extra_msa_stack.tune_chunk_size = True
        self.config.evoformer_stack.tune_chunk_size = True
        self.evoformer.blocks_per_ckpt = self.orig_blocks_per_ckpt

        if self.template_config.enabled:
            self.template_embedder = self.template_embedder.to(torch.float32)

    def opt_settings(self, iter_guide_config):
        precision = str_to_dtype[iter_guide_config.precision]
        self.evoformer = self.evoformer.to(precision)
        self.structure_module = self.structure_module.to(precision)
        self.aux_heads = self.aux_heads.to(precision)
        self.recycling_embedder = self.recycling_embedder.to(precision)
        self.extra_msa_stack = self.extra_msa_stack.to(precision)

        self.globals.offload_inference = True
        self.globals.use_deepspeed_evo_attention = False
        self.config.template.offload_inference = True
        self.config.template.template_pair_stack.tune_chunk_size = False
        self.config.extra_msa.extra_msa_stack.tune_chunk_size = False
        self.config.evoformer_stack.tune_chunk_size = False
        self.evoformer.blocks_per_ckpt = 1

        if self.template_config.enabled:
            self.template_embedder = self.template_embedder.to(precision)

        if self.device_group is None:
            num_gpus = torch.cuda.device_count()
            self.device_group = (
                [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 1 else None
            )
        print("DEVICE GROUP", self.device_group)

    def embed_templates(self, batch, feats, z, pair_mask, templ_dim, inplace_safe):
        if self.globals.is_multimer:
            asym_id = feats["asym_id"]
            multichain_mask_2d = asym_id[..., None] == asym_id[..., None, :]
            template_embeds = self.template_embedder(
                batch,
                z,
                pair_mask.to(dtype=z.dtype),
                templ_dim,
                chunk_size=self.globals.chunk_size,
                multichain_mask_2d=multichain_mask_2d,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )
            feats["template_torsion_angles_mask"] = template_embeds["template_mask"]
        else:
            if self.template_config.offload_templates:
                return embed_templates_offload(
                    self,
                    batch,
                    z,
                    pair_mask,
                    templ_dim,
                    inplace_safe=inplace_safe,
                )
            elif self.template_config.average_templates:
                return embed_templates_average(
                    self,
                    batch,
                    z,
                    pair_mask,
                    templ_dim,
                    inplace_safe=inplace_safe,
                )

            template_embeds = self.template_embedder(
                batch,
                z,
                pair_mask.to(dtype=z.dtype),
                templ_dim,
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        return template_embeds

    def tolerance_reached(self, prev_pos, next_pos, mask, eps=1e-8) -> bool:
        """
        Early stopping criteria based on criteria used in
        AF2Complex: https://www.nature.com/articles/s41467-022-29394-2
        Args:
          prev_pos: Previous atom positions in atom37/14 representation
          next_pos: Current atom positions in atom37/14 representation
          mask: 1-D sequence mask
          eps: Epsilon used in square root calculation
        Returns:
          Whether to stop recycling early based on the desired tolerance.
        """

        def distances(points):
            """Compute all pairwise distances for a set of points."""
            d = points[..., None, :] - points[..., None, :, :]
            return torch.sqrt(torch.sum(d**2, dim=-1))

        if self.config.recycle_early_stop_tolerance < 0:
            return False

        ca_idx = residue_constants.atom_order["CA"]
        sq_diff = (
            distances(prev_pos[..., ca_idx, :]) - distances(next_pos[..., ca_idx, :])
        ) ** 2
        mask = mask[..., None] * mask[..., None, :]
        sq_diff = masked_mean(
            mask=mask, value=sq_diff, dim=list(range(len(mask.shape)))
        )
        diff = torch.sqrt(sq_diff + eps).item()
        return diff <= self.config.recycle_early_stop_tolerance

    def get_optimizer(
        self, params: List[torch.Tensor], iter_guide_config
    ) -> torch.optim.Optimizer:
        if iter_guide_config.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=iter_guide_config.guidance_lr)
        elif iter_guide_config.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr=iter_guide_config.guidance_lr)
        else:
            raise ValueError(f"Optimizer {iter_guide_config.optimizer} not supported")
        return optimizer

    def log_metrics(self, metrics: List[List[str]]):
        print(tabulate(metrics, headers=["Name", "Value"], tablefmt="simple_grid"))

    def entropy_loss(self, z: torch.Tensor) -> torch.Tensor:
        pred_logits = self.aux_heads.distogram(z)
        pred_probs = torch.softmax(pred_logits, dim=-1)

        return entropy(pred_probs).mean()

    def gt_distogram_loss(
        self, z: torch.Tensor, iter_guide_config: dict, tag: str
    ) -> torch.Tensor:
        device = z.device
        pred_logits = self.aux_heads.distogram(z)

        assert iter_guide_config.gt_distances_path is not None
        gt_distances_path = iter_guide_config.gt_distances_path.format(tag=tag)
        print(f"Loading distances from {gt_distances_path}")
        gt_distances = np.load(gt_distances_path)
        if iter_guide_config.gt_mask_path is not None:
            gt_mask_path = iter_guide_config.gt_mask_path.format(tag=tag)
            print(f"Loading mask from {gt_mask_path}")
            gt_mask = np.load(gt_mask_path)
        else:
            gt_mask = np.ones_like(gt_distances)
        gt_distances = torch.as_tensor(gt_distances).to(device)
        gt_mask = torch.as_tensor(gt_mask).to(device).flatten().type(torch.bool)
        gt_distogram = dists_to_distogram(gt_distances)
        n_buckets = pred_logits.shape[-1]
        loss = F.cross_entropy(
            pred_logits.reshape([-1, n_buckets])[gt_mask],
            gt_distogram.flatten()[gt_mask],
        )
        return loss

    def run_evoformer(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        inplace_safe: bool,
        offload_inference: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        func = self.evoformer._forward_offload if offload_inference else self.evoformer
        first_arg = [[m, z]] if offload_inference else [m, z]
        print("DEVICE GROUP", self.device_group)
        extra_kwargs = {"inplace_safe": inplace_safe} if not offload_inference else {}
        new_m, new_z, new_s = func(
            *first_arg,
            msa_mask=msa_mask.to(m.dtype),
            pair_mask=pair_mask.to(z.dtype),
            chunk_size=self.globals.chunk_size,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            use_lma=self.globals.use_lma,
            use_flash=self.globals.use_flash,
            _mask_trans=self.config._mask_trans,
            device_group=self.device_group,
            **extra_kwargs,
        )
        return new_m, new_z, new_s

    def run_structure_module(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        feats: dict,
        device: str,
        inplace_safe: bool,
        offload_inference: bool,
    ) -> dict:
        return self.structure_module(
            {"msa": m, "pair": z, "single": s},
            feats["aatype"].to(device),
            mask=feats["seq_mask"].to(dtype=s.dtype).to(device),
            inplace_safe=inplace_safe,
            _offload_inference=offload_inference,
        )

    def opt(
        self,
        orig_m: torch.Tensor,
        orig_z: torch.Tensor,
        iter_guide_config: dict,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        feats: dict,
        device: str,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, List[dict], List[List[str]], List[torch.Tensor]
    ]:
        precision = str_to_dtype[iter_guide_config.precision]
        torch.set_grad_enabled(True)
        m_param = torch.zeros_like(orig_m, requires_grad=True, device=device)
        z_param = torch.zeros_like(orig_z, requires_grad=True, device=device)
        optimizer = self.get_optimizer([m_param, z_param], iter_guide_config)

        all_distograms = []
        all_sm_outputs = []
        all_metrics = []
        for i in range(iter_guide_config.num_gradient_steps):
            metrics = []
            m_input = orig_m + m_param
            z_input = orig_z + z_param

            print(f"Step {i}")
            start = time.time()
            new_m, new_z, new_s = self.run_evoformer(
                m_input, z_input, msa_mask, pair_mask, False, False
            )
            if self.guide_config.save_distograms:
                with torch.no_grad():
                    distogram = self.aux_heads.distogram(new_z.detach())
                    all_distograms.append(distogram.detach())

            if self.guide_config.log_intermediate_structures:
                print("Running structure module")
                final_sm_outputs = {}
                sm_outputs = self.run_structure_module(
                    new_m, new_z, new_s, feats, device, False, False
                )
                with torch.no_grad():
                    final_sm_outputs["sm"] = {
                        k: sm_outputs[k].detach()
                        for k in [
                            "positions",
                            "frames",
                            "sidechain_frames",
                            "angles",
                            "unnormalized_angles",
                        ]
                    }
                    if hasattr(self.aux_heads, "tm"):
                        tm_logits = self.aux_heads.tm(new_z.detach())
                        tm_score = compute_tm(tm_logits, **self.aux_heads.config.tm)
                    else:
                        tm_score = torch.as_tensor(-1.0)
                    lddt_logits = self.aux_heads.plddt(sm_outputs["single"].detach())
                    plddt = compute_plddt(lddt_logits)
                    print(f"TMSCORE: {tm_score:.3f} PLDDT: {plddt.mean():.3f}")
                    final_sm_outputs["tmscore"] = tm_score.detach()
                    final_sm_outputs["predicted_tm_score"] = tm_score.detach()
                    final_sm_outputs["plddt"] = plddt.detach()
                    final_sm_outputs["true_plddt"] = plddt.mean().detach()
                    all_sm_outputs.append(final_sm_outputs)
                if i == 0:
                    initial_structure = copy.deepcopy(final_sm_outputs)
                    initial_structure = self.postprocess_structure(
                        initial_structure, feats
                    )
                    last_structure = initial_structure
                    curr_structure = initial_structure
                else:
                    curr_structure = copy.deepcopy(final_sm_outputs)
                    curr_structure = self.postprocess_structure(curr_structure, feats)

                    curr_ca = (
                        curr_structure["final_atom_positions"][:, 1]
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    initial_ca = (
                        initial_structure["final_atom_positions"][:, 1]
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    curr_rmsd = full_rmsd(curr_ca, initial_ca)
                    print(f"Current RMSD: {curr_rmsd:.3f}")

            total_loss = torch.tensor(0.0, device=device, dtype=precision)
            if iter_guide_config.entropy_weight is not None:
                entropy_loss = self.entropy_loss(new_z)
                total_loss = (
                    total_loss + iter_guide_config.entropy_weight * entropy_loss
                )
                metrics.append(["Entropy Loss", entropy_loss.item()])

            if iter_guide_config.gt_distogram_weight is not None:
                gt_distogram_loss = self.gt_distogram_loss(
                    new_z, iter_guide_config, tag=feats["tag"]
                )
                total_loss = (
                    total_loss
                    + iter_guide_config.gt_distogram_weight * gt_distogram_loss
                )
                metrics.append(["GT Distogram Loss", gt_distogram_loss.item()])

            self.log_metrics(metrics)
            all_metrics.append(metrics)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Step {i} took {time.time() - start:.3f}s")

        m_param = m_param.detach() + orig_m
        z_param = z_param.detach() + orig_z
        m = m_param.detach()
        z = z_param.detach()
        return m, z, all_sm_outputs, all_metrics, all_distograms

    def postprocess_structure(self, outputs, feats):
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        return outputs

    def iteration(self, feats, prevs, iter_guide_config, _recycle=True):
        precision = str_to_dtype[iter_guide_config.precision]
        # Primary output dictionary
        outputs = {}

        if iter_guide_config.opt:
            self.opt_settings(iter_guide_config)
        else:
            self.reverse_opt_settings()

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if k == "tag":
                continue
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        if self.globals.is_multimer:
            # Initialize the MSA and pair representations
            # m: [*, S_c, N, C_m]
            # z: [*, N, N, C_z]
            m, z = self.input_embedder(feats)
        elif self.seqemb_mode:
            # Initialize the SingleSeq and pair representations
            # m: [*, 1, N, C_m]
            # z: [*, N, N, C_z]
            m, z = self.input_embedder(
                feats["target_feat"], feats["residue_index"], feats["seq_embedding"]
            )
        else:
            # Initialize the MSA and pair representations
            # m: [*, S_c, N, C_m]
            # z: [*, N, N, C_z]
            m, z = self.input_embedder(
                feats["target_feat"],
                feats["residue_index"],
                feats["msa_feat"],
                inplace_safe=inplace_safe,
            )

        if iter_guide_config.opt:
            m = m.to(precision)
            z = z.to(precision)

        # Unpack the recycling embeddings. Removing them from the list allows
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        if m_1_prev is not None and z_prev is not None and x_prev is not None:
            m_1_prev, z_prev, x_prev = (
                m_1_prev.to(precision),
                z_prev.to(precision),
                x_prev.to(precision),
            )

        # Initialize the recycling embeddings, if needs be
        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        pseudo_beta_x_prev = pseudo_beta_fn(feats["aatype"], x_prev, None).to(
            dtype=z.dtype
        )

        # The recycling embedder is memory-intensive, so we offload first
        if self.globals.offload_inference and inplace_safe:
            m = m.cpu()
            z = z.cpu()

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            pseudo_beta_x_prev,
            inplace_safe=inplace_safe,
        )

        del pseudo_beta_x_prev

        if self.globals.offload_inference and inplace_safe:
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)

        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

        # Embed the templates + merge with MSA/pair embeddings
        if self.config.template.enabled:
            template_feats = {
                k: v for k, v in feats.items() if k.startswith("template_")
            }
            for k in template_feats:
                if template_feats[k].dtype == torch.float32:
                    template_feats[k] = template_feats[k].to(precision)
            template_embeds = self.embed_templates(
                template_feats,
                feats,
                z,
                pair_mask.to(dtype=z.dtype),
                no_batch_dims,
                inplace_safe=inplace_safe,
            )

            # [*, N, N, C_z]
            z = add(
                z,
                template_embeds.pop("template_pair_embedding"),
                inplace_safe,
            )

            if "template_single_embedding" in template_embeds:
                # [*, S = S_c + S_t, N, C_m]
                m = torch.cat([m, template_embeds["template_single_embedding"]], dim=-3)

                # [*, S, N]
                if not self.globals.is_multimer:
                    torsion_angles_mask = feats["template_torsion_angles_mask"]
                    msa_mask = torch.cat(
                        [feats["msa_mask"], torsion_angles_mask[..., 2]], dim=-2
                    )
                else:
                    msa_mask = torch.cat(
                        [feats["msa_mask"], template_embeds["template_mask"]],
                        dim=-2,
                    )

        # Embed extra MSA features + merge with pairwise embeddings
        if self.config.extra_msa.enabled:
            if self.globals.is_multimer:
                extra_msa_fn = data_transforms_multimer.build_extra_msa_feat
            else:
                extra_msa_fn = build_extra_msa_feat

            # [*, S_e, N, C_e]
            extra_msa_feat = extra_msa_fn(feats).to(dtype=z.dtype)
            a = self.extra_msa_embedder(extra_msa_feat)

            if iter_guide_config.opt:
                a = a.to(precision)

            if self.globals.offload_inference:
                # To allow the extra MSA stack (and later the evoformer) to
                # offload its inputs, we remove all references to them here
                input_tensors = [a, z]
                del a, z

                # [*, N, N, C_z]
                z = self.extra_msa_stack._forward_offload(
                    input_tensors,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    _mask_trans=self.config._mask_trans,
                )

                del input_tensors
            else:
                # [*, N, N, C_z]
                z = self.extra_msa_stack(
                    a,
                    z,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )

        m, z = m.detach(), z.detach()
        all_distograms = []
        all_metrics = []
        all_sm_outputs = []
        if iter_guide_config.opt:
            m, z, all_sm_outputs, all_metrics, all_distograms = self.opt(
                m, z, iter_guide_config, msa_mask, pair_mask, feats, device
            )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]
        m, z = m.detach(), z.detach()
        if self.globals.offload_inference and not iter_guide_config.opt:
            input_tensors = [m, z]
            del m, z
            m, z, s = self.run_evoformer(
                input_tensors[0],
                input_tensors[1],
                msa_mask,
                pair_mask,
                inplace_safe,
                offload_inference=True,
            )

            del input_tensors
        else:
            m, z, s = self.run_evoformer(
                m,
                z,
                msa_mask,
                pair_mask,
                inplace_safe,
                offload_inference=False,
            )

        with torch.no_grad():
            if self.guide_config.save_distograms:
                all_distograms.append(self.aux_heads.distogram(z.detach()).detach())

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, N, C_z]
        z_prev = outputs["pair"]

        early_stop = False
        if self.globals.is_multimer:
            early_stop = self.tolerance_reached(
                x_prev, outputs["final_atom_positions"], seq_mask
            )

        del x_prev

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]

        all_sm_outputs = [
            self.postprocess_structure(output, feats) for output in all_sm_outputs
        ]

        with torch.no_grad():
            if hasattr(self.aux_heads, "tm"):
                tm_logits = self.aux_heads.tm(outputs["pair"].detach())
                tm_score = compute_tm(tm_logits, **self.aux_heads.config.tm)
            else:
                tm_score = torch.as_tensor(-1.0)
            lddt_logits = self.aux_heads.plddt(outputs["sm"]["single"].detach())
            plddt = compute_plddt(lddt_logits)
            print(f"TMSCORE: {tm_score:.3f} PLDDT: {plddt.mean():.3f}")
            final_outputs = {
                "sm": {
                    k: outputs["sm"][k]
                    for k in [
                        "positions",
                        "frames",
                        "sidechain_frames",
                        "angles",
                        "unnormalized_angles",
                    ]
                }
            }
            for k in ["final_atom_positions", "final_atom_mask", "final_affine_tensor"]:
                final_outputs[k] = outputs[k].detach()
            final_outputs["tmscore"] = tm_score.detach()
            final_outputs["predicted_tm_score"] = tm_score.detach()
            final_outputs["plddt"] = plddt.detach()
            final_outputs["true_plddt"] = plddt.mean().detach()
            all_sm_outputs.append(final_outputs)

        return (
            outputs,
            m_1_prev,
            z_prev,
            x_prev,
            early_stop,
            all_sm_outputs,
            all_metrics,
            all_distograms,
        )

    def _disable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = None
        self.evoformer.blocks_per_ckpt = None

        for b in self.extra_msa_stack.blocks:
            b.ckpt = False

    def _enable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = (
            self.config.template.template_pair_stack.blocks_per_ckpt
        )
        self.evoformer.blocks_per_ckpt = self.config.evoformer_stack.blocks_per_ckpt

        for b in self.extra_msa_stack.blocks:
            b.ckpt = self.config.extra_msa.extra_msa_stack.ckpt

    def forward(self, batch: Dict[str, torch.Tensor], tag: str):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        early_stop = False
        num_recycles = 0
        intermediate_outputs = []
        intermediate_metrics = []
        for cycle_no in range(num_iters):
            iter_guide_config = getattr(self.guide_config, f"iter_{cycle_no}")

            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)
            feats["tag"] = tag

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1) or early_stop
            with torch.set_grad_enabled(False):  # is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                start = time.time()
                (
                    outputs,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    early_stop,
                    all_sm_outputs,
                    all_metrics,
                    all_distograms,
                ) = self.iteration(
                    feats,
                    prevs,
                    iter_guide_config=iter_guide_config,
                    _recycle=(num_iters > 1),
                )
                print(f"Recycling iteration {cycle_no} took {time.time() - start:.3f}s")

                if self.guide_config.info_dir is not None:
                    # Save the outputs of the current recycling iteration
                    keys = [
                        "sm",
                        "final_atom_positions",
                        "final_atom_mask",
                        "plddt",
                        "true_plddt",
                        "tmscore",
                        "predicted_tm_score",
                    ]
                    new_sm_outputs = [
                        {k: outputs[k] for k in keys} for outputs in all_sm_outputs
                    ]
                    all_sm_outputs = [
                        {
                            k: outputs[k].cpu()
                            if isinstance(outputs[k], torch.Tensor)
                            else outputs[k]
                            for k in outputs
                        }
                        for outputs in new_sm_outputs
                    ]
                    intermediate_outputs.extend(all_sm_outputs)
                    intermediate_metrics.extend(all_metrics)

                    if self.guide_config.save_distograms:
                        save_distogram_path = os.path.join(
                            self.guide_config.info_dir,
                            f"{tag}_distograms_{cycle_no}.pt",
                        )
                        all_distograms = [d.cpu() for d in all_distograms]
                        all_distograms = torch.stack(all_distograms, dim=0)
                        print(f"Saving distograms to {save_distogram_path}")
                        torch.save(all_distograms, save_distogram_path)

                num_recycles += 1

                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev
                else:
                    break

        outputs["num_recycles"] = torch.tensor(
            num_recycles, device=feats["aatype"].device
        )

        if "asym_id" in batch:
            outputs["asym_id"] = feats["asym_id"]

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        return outputs, intermediate_outputs, intermediate_metrics
