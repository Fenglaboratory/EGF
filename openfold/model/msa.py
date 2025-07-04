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
from functools import partial
import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from openfold.model.primitives import (
    Linear,
    LayerNorm,
    Attention,
    GlobalAttention,
    _attention_chunked_trainable,
)
from openfold.utils.checkpointing import get_checkpoint_fn
from openfold.utils.chunk_utils import chunk_layer
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(self.c_z, self.no_heads, bias=False, init="normal")

        self.mha = Attention(
            self.c_in,
            self.c_in,
            self.c_in,
            self.c_hidden,
            self.no_heads,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        biases: Optional[List[torch.Tensor]],
        chunk_size: int,
        use_memory_efficient_kernel: bool,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        use_flash: bool,
        flash_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        def fn(m, biases, flash_mask):
            m = self.layer_norm_m(m)
            return self.mha(
                q_x=m,
                kv_x=m,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=flash_mask,
            )

        inputs = {"m": m}
        if biases is not None:
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)
        if use_flash and flash_mask is not None:
            inputs["flash_mask"] = flash_mask
        else:
            fn = partial(fn, flash_mask=None)

        return chunk_layer(
            fn, inputs, chunk_size=chunk_size, no_batch_dims=len(m.shape[:-2])
        )

    def _prep_inputs(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(
                m.shape[:-3] + (n_seq, n_res),
            )

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        if (
            self.pair_bias
            and z is not None
            and self.layer_norm_z is not None  # For the
            and self.linear_z is not None  # benefit of  # TorchScript
        ):
            chunks = []

            for i in range(0, z.shape[-3], 256):
                z_chunk = z[..., i : i + 256, :, :]

                # [*, N_res, N_res, C_z]
                z_chunk = self.layer_norm_z(z_chunk)

                # [*, N_res, N_res, no_heads]
                z_chunk = self.linear_z(z_chunk)

                chunks.append(z_chunk)

            z = torch.cat(chunks, dim=-3)

            # [*, 1, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    @torch.jit.ignore
    def _chunked_msa_attn(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_logits: int,
        checkpoint: bool,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        MSA attention with training-time chunking of the softmax computation.
        Saves memory in the extra MSA stack. Probably obviated by our fused
        attention kernel, which is now used by default.
        """
        MSA_DIM = -4

        def _get_qkv(m, z):
            m, mask_bias, z = self._prep_inputs(m, z, mask, inplace_safe=inplace_safe)
            m = self.layer_norm_m(m)
            q, k, v = self.mha._prep_qkv(m, m)
            return m, q, k, v, mask_bias, z

        checkpoint_fn = get_checkpoint_fn()

        if torch.is_grad_enabled() and checkpoint:
            m, q, k, v, mask_bias, z = checkpoint_fn(_get_qkv, m, z)
        else:
            m, q, k, v, mask_bias, z = _get_qkv(m, z)

        o = _attention_chunked_trainable(
            query=q,
            key=k,
            value=v,
            biases=[mask_bias, z],
            chunk_size=chunk_logits,
            chunk_dim=MSA_DIM,
            checkpoint=checkpoint,
        )

        if torch.is_grad_enabled() and checkpoint:
            # Storing an additional m here is far from ideal
            m = checkpoint_fn(self.mha._wrap_up, o, m)
        else:
            m = self.mha._wrap_up(o, m)

        return m

    def forward(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _chunk_logits: Optional[int] = None,
        _checkpoint_chunks: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the
                cost of slower execution. Chunking is not performed by default.

        """
        if _chunk_logits is not None:
            return self._chunked_msa_attn(
                m=m,
                z=z,
                mask=mask,
                chunk_logits=_chunk_logits,
                checkpoint=_checkpoint_chunks,
                inplace_safe=inplace_safe,
            )

        if use_flash:
            assert z is None
            biases = None
        else:
            m, mask_bias, z = self._prep_inputs(m, z, mask, inplace_safe=inplace_safe)

            biases = [mask_bias]
            if z is not None:
                biases.append(z)

        if chunk_size is not None:
            m = self._chunk(
                m,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=mask,
            )
        else:
            m = self.layer_norm_m(m)
            m = self.mha(
                q_x=m,
                kv_x=m,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=mask,
            )

        return m


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSARowAttentionWithPairBias, self).__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.

    By rights, this should also be a subclass of MSAAttention. Alas,
    most inheritance isn't supported by TorchScript.
    """

    def __init__(self, c_m, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                MSA channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSAColumnAttention, self).__init__()

        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self._msa_att = MSAAttention(
            c_in=c_m,
            c_hidden=c_hidden,
            no_heads=no_heads,
            pair_bias=False,
            c_z=None,
            inf=inf,
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the
                cost of slower execution. Chunking is not performed by default.
        """
        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        m = self._msa_att(
            m,
            mask=mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
        )

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        return m


class MSAColumnGlobalAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        inf=1e9,
        eps=1e-10,
    ):
        super(MSAColumnGlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.layer_norm_m = nn.LayerNorm(c_in)

        self.global_attention = GlobalAttention(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf,
            eps=eps,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        use_lma: bool = False,
    ) -> torch.Tensor:
        mha_input = {
            "m": m,
            "mask": mask,
        }

        def fn(m, mask):
            m = self.layer_norm_m(m)
            return self.global_attention(m, mask, use_lma=use_lma)

        return chunk_layer(
            fn,
            mha_input,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ) -> torch.Tensor:
        n_seq, n_res, c_in = m.shape[-3:]

        if mask is None:
            # [*, N_seq, N_res]
            mask = torch.ones(
                m.shape[:-1],
                dtype=m.dtype,
                device=m.device,
            ).detach()

        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size, use_lma=use_lma)
        else:
            m = self.layer_norm_m(m)
            m = self.global_attention(m=m, mask=mask, use_lma=use_lma)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m
