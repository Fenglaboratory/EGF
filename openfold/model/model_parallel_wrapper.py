from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os


def to_device(x, device):
    if isinstance(x, torch.Tensor) or isinstance(x, nn.Module):
        return x.to(device)
    else:
        return x


class AutoModelParallelBlock(nn.Module):
    def __init__(self, block, device, out_device=None):
        super().__init__()
        self.block = block
        self.device = device
        self.block = to_device(self.block, self.device)
        self.out_device = out_device if out_device is not None else self.device

    def forward(self, *args, **kwargs):
        def get_tensors(r):
            results = []
            for x in r:
                if isinstance(x, torch.Tensor):
                    results.append(x)
                else:
                    results.extend(get_tensors(x))
            return results

        largs = get_tensors(args)
        lkwargs = kwargs
        largs = [to_device(x, device=self.device) for x in largs]
        lkwargs = {
            key: to_device(value, device=self.device) for key, value in lkwargs.items()
        }

        with torch.set_grad_enabled(True):
            outputs = self.block(*largs, **lkwargs)
        outputs = [to_device(x, device=self.out_device) for x in outputs]
        return outputs[0], outputs[1]


class AutoModelParallel(nn.Module):
    def __init__(self, blocks, device_group):
        super().__init__()
        self.blocks = blocks
        self.device_group = device_group
        if len(device_group) == 1:
            self.indices = [0] * len(self.blocks)
        elif len(device_group) == 2:
            # This works best
            self.indices = [0] * 8 + [1] * (len(self.blocks) - 8)
        else:
            blocks_per_device = len(self.blocks) // len(device_group)
            self.indices = [
                min(i // blocks_per_device, len(device_group) - 1)
                for i in range(len(self.blocks))
            ]
        self.blocks = nn.ModuleList(
            [
                AutoModelParallelBlock(
                    block,
                    self.device_group[self.indices[i]],
                    out_device=self.device_group[0],
                )
                for i, block in enumerate(self.blocks)
            ]
        )
        for i in range(len(self.blocks)):
            if i == len(self.blocks) - 1:
                self.blocks[i].out_device = self.device_group[0]
            else:
                self.blocks[i].out_device = self.blocks[i].device

    def __getitem__(self, idx):
        return self.blocks[idx]

    def __len__(self):
        return len(self.blocks)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "AutoModelParallel cannot replace your forward logic. Use the .blocks attribute or index the model."
        )
