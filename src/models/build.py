#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(gpu_id=None): #cfg, gpu_id=None):
    """
    Builds the model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in mvit/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            torch.cuda.device_count()>=1
        ), "Cannot use more GPU devices than available"
    else:
        # assert (
        #     cfg.NUM_GPUS == 0
        # ),
        "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = "MViT"
    model = MODEL_REGISTRY.get(name)#(cfg)

    #if cfg.NUM_GPUS:
    if gpu_id is None:
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = gpu_id
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    # if cfg.NUM_GPUS > 1:
    #     # Make model replica operate on the current device
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         module=model, device_ids=[cur_device], output_device=cur_device
    #     )
    return model
