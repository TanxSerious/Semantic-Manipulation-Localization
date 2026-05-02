# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Optional, Tuple, Type

from .common import LayerNorm2d

import math
import copy
from functools import partial
from dataclasses import dataclass, field
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
try:
    from models.sam3.sam import mamba_conv
    _mamba_available = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Failed to import mamba_conv, MambaConv features will be disabled. Error: {e}")
    mamba_conv = None
    _mamba_available = False

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = True,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = True,
        mamba_conv_d_model: int = 32,
        mamba_conv_n_layer: int = 3,
        mamba_conv_edge_n_layer: int = 3,# FusedDualScanMamba 不再单独用此参数
        use_mask_mamba_refine: bool = True,
        use_mask_path: bool = True,    # 新增
        use_edge_path: bool = True,    # 新增
        mamba_attn_layer_idx: list = [2],     # 例如 [2] 表示第2层用注意力
        mamba_attn_cfg: dict = {"num_heads": 4},           # 例如 {"num_heads": 4}
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )

        '''
        # all mamba blocks
        self.mamba_refine = MambaBlock(
            dim=transformer_dim // 8,
            kernel_size=31,
            expansion=2,
            dropout=0.0,
        )
        '''
        '''
        self.mamba_for_mask = MambaBlock(
            dim=1,
            kernel_size=15,
            expansion=1,
            dropout=0.0,
        )
        self.mamba_gamma = nn.Parameter(torch.zeros(1))
        

        mask_cfg = mamba_conv.MambaConfig(
                d_model=mamba_conv_d_model,
                d_intermediate=0,
                n_layer=mamba_conv_n_layer,
                ssm_cfg={"layer": "Mamba1", "d_state": 16},
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=False,
            )
        self.mask_mamba_conv_in = nn.Conv2d(1, mask_cfg.d_model, kernel_size=1, stride=1)
        self.mask_mamba_conv = mamba_conv.MixerModel(**mask_cfg.__dict__)
        self.mask_mamba_conv_out = nn.Conv2d(mask_cfg.d_model, 1, kernel_size=1, stride=1)
        self.mask_mamba_gamma = nn.Parameter(torch.zeros(1))
        '''
        self.use_mask_mamba_refine = use_mask_mamba_refine
        if self.use_mask_mamba_refine:
            if mamba_attn_layer_idx is None:
                mamba_attn_layer_idx = []
            if mamba_attn_cfg is None:
                mamba_attn_cfg = {}
            self.mask_mamba_refine = mamba_conv.FusedDualScanMamba(
                d_model=mamba_conv_d_model,
                n_layer=mamba_conv_n_layer,
                patch_size=4,
                d_state=16,
                dwconv_kernel=3,             # 2D局部混合的卷积核大小
                dropout=0.0,
                use_mask_path=use_mask_path,
                use_edge_path=use_edge_path,
                attn_layer_idx=mamba_attn_layer_idx,
                attn_cfg=mamba_attn_cfg,
            )
        '''
        if self.use_mask_mamba_refine:
            self.mask_mamba_refine = mamba_conv.DualPathPatchMamba(
                d_model=mamba_conv_d_model,
                n_layer=mamba_conv_n_layer,
                patch_size=4,
                d_state=16,
                dwconv_kernel=7,
                dropout=0.0,
                edge_n_layer=mamba_conv_edge_n_layer,
                use_mask_path=use_mask_path,      # 传递开关
                use_edge_path=use_edge_path,      # 传递开关
            )
        '''


        # all mamba blocks end


        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits, edge_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits, edge_pred
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # buling mamba block1
        # upscaled_embedding = self.mamba_refine(upscaled_embedding)
        # buling mamba block1 end

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

    
        '''
        # buling mamba block2
        B, K, H, W = masks.shape
        m = masks.view(B * K, 1, H, W)
        m = m + self.mamba_gamma * self.mamba_for_mask(m)
        masks = m.view(B, K, H, W)
        # buling mamba block2 end
        '''
        '''
        # buling mamba block3
        B, K, H, W = masks.shape
        m = masks.view(B * K, 1, H, W)
        mm = self.mask_mamba_conv_in(m) # B*K, d_model, H, W
        bk, dm, h_m, w_m = mm.shape
        mm = mm.flatten(2).transpose(1, 2) # B*K, H*W, d_model
        mm = self.mask_mamba_conv(mm) 
        mm = mm.transpose(1, 2).reshape(bk, dm, h_m, w_m)
        mm = self.mask_mamba_conv_out(mm)
        m = m + self.mask_mamba_gamma * mm
        masks = m.view(B, K, H, W)
        # buling mamba block3 end
        '''
        '''
        # buling mamba block4
        B, K, H, W = masks.shape
        m = masks.view(B * K, 1, H, W)
        mm = self.mask_mamba_conv_in(m)
        bk, dm, h_m, w_m = mm.shape
        mm = mm.flatten(2).transpose(1, 2)
        mm = self.mask_mamba_conv(mm)
        mm = mm.transpose(1, 2).reshape(bk, dm, h_m, w_m)
        mm = self.mask_mamba_conv_out(mm)
        m = m + self.mask_mamba_gamma * mm
        masks = m.view(B, K, H, W)
        # buling mamba block4 end
        
        # buling mamba block5
        B, K, H, W = masks.shape
        m = masks.view(B * K, 1, H, W)
        mm_in = self.mask_mamba_conv_in(m)
        bk, dm, h_m, w_m = mm_in.shape
        mm_seq = mm_in.flatten(2).transpose(1, 2)
        mm_seq = self.mask_mamba_conv(mm_seq)
        mm_out = mm_seq.transpose(1, 2).reshape(bk, dm, h_m, w_m)
        mm_final = self.mask_mamba_conv_out(mm_out)
        m = m + self.mask_mamba_gamma * mm_final
        masks = m.view(B, K, H, W)
        # buling mamba block5 end
        '''
    
        # buling mamba block — Dual-Path Mamba refinement (mask + edge)
        if self.use_mask_mamba_refine:
            B, K, H, W = masks.shape
            m = masks.reshape(B * K, 1, H, W)
            m, edge_pred = self.mask_mamba_refine(m, return_edge=True)
            masks = m.reshape(B, K, H, W)
            # average edge across mask tokens → per-image edge map
            if edge_pred is not None:
                edge_pred = edge_pred.reshape(B, K, 1, H, W).mean(dim=1)  # (B, 1, H, W)
        else:
            edge_pred = None
        # buling mamba block end

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits, edge_pred
    
    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=31, expansion=2, dropout=0.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.dim = dim
        self.kernel_size = kernel_size
        self.expansion = expansion

        self.norm = nn.LayerNorm(dim)

        self.dw_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
            bias=False,
        )

        hidden_dim = int(dim * expansion)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.proj = nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_seq)
        x_conv = self.dw_conv(x_norm.transpose(1, 2)).transpose(1, 2)
        x_mlp = self.mlp(x_conv)
        out_seq = x_seq + x_mlp
        out = out_seq.transpose(1, 2).reshape(b, c, h, w)
        return out
    
    
