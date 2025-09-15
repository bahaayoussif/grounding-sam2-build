# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# Deformable DETR
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/multi_scale_deform_attn.py
# ------------------------------------------------------------------------------------------------

import math
import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

# Try to import the compiled extension. Fall back gracefully if missing.
HAVE_CUSTOM_KERNEL = False
FORCE_PYTORCH_MSDA = os.getenv("GROUNDINGDINO_FORCE_PYTORCH_MSDA", "0") == "1"
try:
    from groundingdino import _C  # compiled C++/CUDA/CPU ops
    HAVE_CUSTOM_KERNEL = True
except Exception:
    _C = None
    warnings.warn(
        "Failed to load GroundingDINO custom C++ ops for Multi-Scale Deformable Attention. "
        "Falling back to PyTorch implementation (slower). "
        "You can force this fallback via GROUNDINGDINO_FORCE_PYTORCH_MSDA=1."
    )

# ----------------------- helpers -----------------------

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


# ----------------------- custom Function (uses _C if available) -----------------------

class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        if not HAVE_CUSTOM_KERNEL or _C is None:
            # Should not be called when kernel is missing; caller should use PyTorch fallback instead.
            raise RuntimeError(
                "Custom kernel _C is not available; use PyTorch fallback path instead."
            )

        ctx.im2col_step = im2col_step
        output = _C.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _C.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


# ----------------------- pure PyTorch fallback -----------------------

@torch.no_grad()
def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch (grid_sample) implementation of multi-scale deformable attention.
    Works on CPU/MPS/CUDA but is slower than the custom kernel.
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads2, num_levels, num_points, _ = sampling_locations.shape
    assert num_heads == num_heads2, "num_heads mismatch between value and sampling_locations"

    # Split per level
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes.tolist()], dim=1)
    sampling_grids = 2 * sampling_locations - 1  # normalize to [-1, 1] for grid_sample
    sampling_value_list = []

    for level, (H_, W_) in enumerate(value_spatial_shapes.tolist()):
        # value_l_: (bs, H_*W_, num_heads, embed_dims) -> (bs*num_heads, embed_dims, H_, W_)
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # sampling_grid_l_: (bs, num_queries, num_heads, num_points, 2)
        #                 -> (bs, num_heads, num_queries, num_points, 2)
        #                 -> (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # (bs*num_heads, embed_dims, num_queries, num_points)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # attention: (bs, num_queries, num_heads, num_levels, num_points)
    # -> (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )

    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# ----------------------- Module -----------------------

class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module used in Deformable-DETR."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        batch_first: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        head_dim = embed_dim // num_heads

        self.batch_first = batch_first

        if not _is_power_of_2(head_dim):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn so that each head dim is a power of 2."
            )

        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def _reset_parameters(self):
        return self.init_weights()

    def init_weights(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def freeze_sampling_offsets(self):
        print("Freeze sampling offsets")
        self.sampling_offsets.weight.requires_grad = False
        self.sampling_offsets.bias.requires_grad = False

    def freeze_attention_weights(self):
        print("Freeze attention weights")
        self.attention_weights.weight.requires_grad = False
        self.attention_weights.bias.requires_grad = False

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward Function of MultiScaleDeformableAttention"""

        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # (num_query, bs, dim) -> (bs, num_query, dim)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs2, num_value, _ = value.shape
        assert bs == bs2
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            # (x, y)
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            # (x, y, w, h)
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        # Decide which path to use
        use_pytorch_fallback = FORCE_PYTORCH_MSDA or (not HAVE_CUSTOM_KERNEL) or (_C is None)

        if use_pytorch_fallback:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        else:
            # Custom kernel path (fast)
            # ms_deform_attn expects contiguous tensors
            output = MultiScaleDeformableAttnFunction.apply(
                value.contiguous(),
                spatial_shapes.contiguous(),
                level_start_index.contiguous(),
                sampling_locations.contiguous(),
                attention_weights.contiguous(),
                self.im2col_step,
            )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


# ----------------------- dummy helpers (unchanged) -----------------------

def create_dummy_class(klass, dependency, message=""):
    err = f"Cannot import '{dependency}', therefore '{klass}' is not available."
    if message:
        err = err + " " + message

    class _DummyMetaClass(type):
        def __getattr__(_, __):  # noqa: B902
            raise ImportError(err)

    class _Dummy(object, metaclass=_DummyMetaClass):
        def __init__(self, *args, **kwargs):
            raise ImportError(err)

    return _Dummy


def create_dummy_func(func, dependency, message=""):
    if isinstance(dependency, (list, tuple)):
        dependency = ",".join(dependency)
    err = f"Cannot import '{dependency}', therefore '{func}' is not available."
    if message:
        err = err + " " + message

    def _dummy(*args, **kwargs):
        raise ImportError(err)

    return _dummy