import math
import copy
from functools import partial
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states
    
@dataclass
class MambaConfig:

    d_model: int = 64
    d_intermediate: int = 0
    n_layer: int = 3
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True

class MambaConv(nn.Module):
    def __init__(self, cnn_patch_size, cnn_step):
        super().__init__()
        out_channels = 16
        kernel_size = (5, 64)
        batch_norm = True
        dropout_p = None
        subject_num = 16

        self.conv = nn.Sequential(
            # input: batch_size * 1 * patch_size * 64
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size[0] // 2, 0)),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
        )
        self.pool_mean = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels),
            nn.ReLU(),
        )
        self.linear = nn.Linear(4 * out_channels, 2 + subject_num)

        self.cnn_patch_size = cnn_patch_size
        self.cnn_step = cnn_step
        self.pad = False

        self.mamba_cfg = MambaConfig()
        d_model = self.mamba_cfg.d_model
        input_dim = 64
        output_dim = 2
        self.mamba_cfg.ssm_cfg = {
            "layer": "Mamba1",
            "d_state": 16,
        }
        self.mamba_input = nn.Linear(input_dim, d_model)
        self.mamba = MixerModel(**self.mamba_cfg.__dict__)
        self.mamba_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, d_model)),
            nn.Flatten()
        )
        self.mamba_head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: batch_size * patch_size * 64
        patch_size = x.size(1)
        x = x.unsqueeze(1)
        cnn_states = []

        if self.pad:
            patch_size += self.cnn_patch_size
            x = F.pad(x, (0, 0, self.cnn_patch_size // 2, self.cnn_patch_size // 2))

        for i in range(0, patch_size-self.cnn_patch_size+1, self.cnn_step):
            y = self.conv(x[:, :, i:i+self.cnn_patch_size])
            y = self.pool_mean(y)
            hid = self.fc(y)
            cnn_states.append(hid)
        y = torch.stack(cnn_states, dim=1)
        mamba_y = y
        mamba_y = self.mamba_input(mamba_y)
        mamba_y = self.mamba(mamba_y)
        mamba_y = self.mamba_pool(mamba_y)
        prediction = self.mamba_head(mamba_y)

        return prediction

@dataclass
class MambaRefineConfig:
    """Separate config name to avoid collision with mamba_ssm.MambaConfig"""
    d_model: int = 64
    d_intermediate: int = 0
    n_layer: int = 3
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True


class BidirectionalPatchMamba(nn.Module):
    """
    Bidirectional Patch-Mamba for 2D single-channel mask refinement.
    
    Reference:
    - VMamba (Liu et al., ICLR 2024): Patch tokenization + SSM
    - Vision Mamba (Zhu et al., 2024): Bidirectional SSM scanning
    - LayerScale (Touvron et al., 2021): Learnable residual scaling
    """
    
    def __init__(
        self,
        d_model: int = 32,
        n_layer: int = 2,
        patch_size: int = 4,
        d_state: int = 16,
        dwconv_kernel: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4 for 2D sincos PE"
        self.patch_size = patch_size
        self.d_model = d_model
        
        # ---- Patch Embedding ----
        self.patch_embed = nn.Conv2d(
            1, d_model, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.embed_norm = nn.LayerNorm(d_model)  # Fix #3: normalize after patch embedding
        
        # ---- Patch Unembedding ----
        self.patch_unembed = nn.ConvTranspose2d(
            d_model, 1, kernel_size=patch_size, stride=patch_size, bias=True
        )
        
        # ---- 2D Positional Encoding (lazy cache) ----
        self._pos_enc_cache = {}
        
        # ---- Pre-Mamba Local Mixing (DWConv) ----  Fix #7: add local mixing before Mamba too
        assert dwconv_kernel % 2 == 1, "dwconv_kernel must be odd"
        self.pre_dwconv = nn.Conv1d(
            d_model, d_model,
            kernel_size=dwconv_kernel,
            padding=dwconv_kernel // 2,
            groups=d_model,
            bias=False,
        )
        self.pre_norm = nn.LayerNorm(d_model)
        self.pre_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        
        # ---- Forward Mamba ----
        fwd_cfg = MambaRefineConfig(
            d_model=d_model, d_intermediate=0, n_layer=n_layer,
            ssm_cfg={"layer": "Mamba1", "d_state": d_state},
            rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        )
        self.mamba_fwd = MixerModel(**fwd_cfg.__dict__)
        
        # ---- Backward Mamba ----
        bwd_cfg = MambaRefineConfig(
            d_model=d_model, d_intermediate=0, n_layer=n_layer,
            ssm_cfg={"layer": "Mamba1", "d_state": d_state},
            rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        )
        self.mamba_bwd = MixerModel(**bwd_cfg.__dict__)
        
        # ---- Bidirectional Fusion Gate ----
        self.fuse_gate = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # ---- Post-Mamba Local Mixing (DWConv + FFN) ----
        self.post_dwconv = nn.Conv1d(
            d_model, d_model,
            kernel_size=dwconv_kernel,
            padding=dwconv_kernel // 2,
            groups=d_model,
            bias=False,
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.post_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        
        # ---- Gated Residual (gamma=0 init → identity at start) ----
        self.gamma = nn.Parameter(torch.zeros(1))


    def _get_2d_sincos_pos_enc(self, h, w, d_model, device, dtype):
        """2D sinusoidal PE: half dims for row, half for column. Returns (1, h*w, d_model)."""
        cache_key = (h, w, d_model, device)
        if cache_key in self._pos_enc_cache:
            return self._pos_enc_cache[cache_key].to(dtype=dtype)
        
        half_d = d_model // 2
        row_pos = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
        col_pos = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1)
        dim_idx = torch.arange(0, half_d, 2, device=device, dtype=torch.float32)
        freq = 1.0 / (10000.0 ** (dim_idx / half_d))
        
        row_enc = torch.cat([torch.sin(row_pos * freq), torch.cos(row_pos * freq)], dim=-1)
        col_enc = torch.cat([torch.sin(col_pos * freq), torch.cos(col_pos * freq)], dim=-1)
        
        row_enc = row_enc.unsqueeze(1).expand(-1, w, -1)
        col_enc = col_enc.unsqueeze(0).expand(h, -1, -1)
        pos_enc = torch.cat([row_enc, col_enc], dim=-1).reshape(1, h * w, d_model)
        
        self._pos_enc_cache[cache_key] = pos_enc
        return pos_enc.to(dtype=dtype)
    
    def forward(self, x, return_edge=False):
        """
        Args:  x: (B, 1, H, W) single-channel mask logits
        Returns: (B, 1, H, W) refined mask logits
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected single-channel input, got {C}"
        
        # Step 1: Patch Embedding + Norm
        tokens_2d = self.patch_embed(x)                          # (B, d, H/p, W/p)
        _, d, h_p, w_p = tokens_2d.shape
        tokens_seq = tokens_2d.flatten(2).transpose(1, 2)        # (B, L, d)
        tokens_seq = self.embed_norm(tokens_seq)                  # Fix #3
        
        # Step 2: Add 2D Positional Encoding
        pos_enc = self._get_2d_sincos_pos_enc(h_p, w_p, d, x.device, x.dtype)
        tokens_seq = tokens_seq + pos_enc
        
        # Step 3: Pre-Mamba Local Mixing (local spatial context for Mamba input)
        pre_local = self.pre_dwconv(tokens_seq.transpose(1, 2)).transpose(1, 2)
        pre_local = self.pre_norm(pre_local)
        tokens_seq = tokens_seq + self.pre_ffn(pre_local)
        
        # Step 4: Bidirectional Mamba
        out_fwd = self.mamba_fwd(tokens_seq)
        out_bwd = self.mamba_bwd(tokens_seq.flip(1))
        out_bwd = out_bwd.flip(1)
        
        gate = torch.sigmoid(self.fuse_gate)
        out_seq = gate * out_fwd + (1.0 - gate) * out_bwd
        
        # Step 5: Post-Mamba Local Mixing
        post_local = self.post_dwconv(out_seq.transpose(1, 2)).transpose(1, 2)
        post_local = self.post_norm(post_local)
        out_seq = out_seq + self.post_ffn(post_local)
        
        # Step 6: Reshape and unembed
        out_2d = out_seq.transpose(1, 2).reshape(B, d, h_p, w_p)
        delta = self.patch_unembed(out_2d)
        if delta.shape[-2:] != x.shape[-2:]:
            delta = F.interpolate(delta, size=(H, W), mode='bilinear', align_corners=False)
        
        # Step 7: Gated Residual
        return x + self.gamma * delta


class DualPathPatchMamba(nn.Module):
    """
    Dual-Path Patch-Mamba: Mask Mamba + Edge Mamba with Cross-Gated Fusion.

    Architecture:
      - Shared patch embedding + 2D sincos PE + pre-mixing
      - Path 1 (Mask Mamba):  Bidirectional Mamba for mask region features
      - Path 2 (Edge Mamba):  Bidirectional Mamba for edge features,
                              with conv head directly supervised by edge GT
      - Cross-Gated Fusion:   Channel-wise gating (not simple addition)
                              learns per-channel importance from both paths
      - Post-mixing + gated residual

    The edge branch learns edge-specific features at the feature level
    through direct edge supervision (no Sobel), while the mask branch
    focuses on region-level features. The cross-gated fusion adaptively
    combines both paths.

    References:
      - VMamba  (Liu et al., ICLR 2024): Patch tokenization + SSM
      - Vision Mamba (Zhu et al., 2024): Bidirectional SSM scanning
    """

    def __init__(
        self,
        d_model: int = 32,
        n_layer: int = 2,
        patch_size: int = 4,
        d_state: int = 16,
        dwconv_kernel: int = 7,
        dropout: float = 0.0,
        edge_n_layer: int = 1,
        use_mask_path: bool = True,   # 消融开关
        use_edge_path: bool = True,   # 消融开关
    ):
        super().__init__()
        assert use_mask_path or use_edge_path, "At least one path must be enabled"
        self.use_mask_path = use_mask_path
        self.use_edge_path = use_edge_path
        # ... 其余保持不变 ...
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4 for 2D sincos PE"
        self.patch_size = patch_size
        self.d_model = d_model

        # ==================== Shared Front-End ====================
        # ---- Patch Embedding ----
        self.patch_embed = nn.Conv2d(
            1, d_model, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.embed_norm = nn.LayerNorm(d_model)

        # ---- 2D Positional Encoding (lazy cache) ----
        self._pos_enc_cache = {}

        # ---- Pre-Mamba Local Mixing (shared, provides local context) ----
        assert dwconv_kernel % 2 == 1, "dwconv_kernel must be odd"
        self.pre_dwconv = nn.Conv1d(
            d_model, d_model,
            kernel_size=dwconv_kernel,
            padding=dwconv_kernel // 2,
            groups=d_model,
            bias=False,
        )
        self.pre_norm = nn.LayerNorm(d_model)
        self.pre_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # ==================== Path 1: Mask Mamba (bidirectional) ====================
        mask_fwd_cfg = MambaRefineConfig(
            d_model=d_model, d_intermediate=0, n_layer=n_layer,
            ssm_cfg={"layer": "Mamba1", "d_state": d_state},
            rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        )
        self.mask_mamba_fwd = MixerModel(**mask_fwd_cfg.__dict__)

        mask_bwd_cfg = MambaRefineConfig(
            d_model=d_model, d_intermediate=0, n_layer=n_layer,
            ssm_cfg={"layer": "Mamba1", "d_state": d_state},
            rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        )
        self.mask_mamba_bwd = MixerModel(**mask_bwd_cfg.__dict__)
        self.mask_bidir_gate = nn.Parameter(torch.zeros(1, 1, d_model))

        # ==================== Path 2: Edge Mamba (bidirectional) ====================
        edge_fwd_cfg = MambaRefineConfig(
            d_model=d_model, d_intermediate=0, n_layer=edge_n_layer,
            ssm_cfg={"layer": "Mamba1", "d_state": d_state},
            rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        )
        self.edge_mamba_fwd = MixerModel(**edge_fwd_cfg.__dict__)

        edge_bwd_cfg = MambaRefineConfig(
            d_model=d_model, d_intermediate=0, n_layer=edge_n_layer,
            ssm_cfg={"layer": "Mamba1", "d_state": d_state},
            rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        )
        self.edge_mamba_bwd = MixerModel(**edge_bwd_cfg.__dict__)
        self.edge_bidir_gate = nn.Parameter(torch.zeros(1, 1, d_model))

        # ---- Edge Head: conv layers → edge map (supervised by edge GT) ----
        self.edge_head = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2,
                               kernel_size=patch_size, stride=patch_size, bias=True),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model // 4, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(d_model // 4),
            nn.GELU(),
            nn.Conv2d(d_model // 4, 1, kernel_size=1, bias=True),
        )

        # ==================== Cross-Gated Fusion ====================
        # Channel-wise gating: gate = σ(W·[mask_proj ∥ edge_proj])
        # fused = gate ⊙ mask_proj + (1−gate) ⊙ edge_proj
        self.fusion_mask_proj = nn.Linear(d_model, d_model)
        self.fusion_edge_proj = nn.Linear(d_model, d_model)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        # ==================== Shared Back-End ====================
        # ---- Post-Mamba Local Mixing ----
        self.post_dwconv = nn.Conv1d(
            d_model, d_model,
            kernel_size=dwconv_kernel,
            padding=dwconv_kernel // 2,
            groups=d_model,
            bias=False,
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.post_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # ---- Patch Unembedding ----
        self.patch_unembed = nn.ConvTranspose2d(
            d_model, 1, kernel_size=patch_size, stride=patch_size, bias=True
        )

        # ---- Gated Residual (gamma=0 init → identity at start) ----
        self.gamma = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #
    #  Positional Encoding
    # ------------------------------------------------------------------ #
    def _get_2d_sincos_pos_enc(self, h, w, d_model, device, dtype):
        cache_key = (h, w, d_model, device)
        if cache_key in self._pos_enc_cache:
            return self._pos_enc_cache[cache_key].to(dtype=dtype)

        half_d = d_model // 2
        row_pos = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
        col_pos = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1)
        dim_idx = torch.arange(0, half_d, 2, device=device, dtype=torch.float32)
        freq = 1.0 / (10000.0 ** (dim_idx / half_d))

        row_enc = torch.cat([torch.sin(row_pos * freq), torch.cos(row_pos * freq)], dim=-1)
        col_enc = torch.cat([torch.sin(col_pos * freq), torch.cos(col_pos * freq)], dim=-1)

        row_enc = row_enc.unsqueeze(1).expand(-1, w, -1)
        col_enc = col_enc.unsqueeze(0).expand(h, -1, -1)
        pos_enc = torch.cat([row_enc, col_enc], dim=-1).reshape(1, h * w, d_model)

        self._pos_enc_cache[cache_key] = pos_enc
        return pos_enc.to(dtype=dtype)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, x, return_edge=False):
        """
        Args:
            x: (B, 1, H, W)  single-channel mask logits
            return_edge: whether to return edge prediction for supervision
        Returns:
            refined: (B, 1, H, W) refined mask logits
            edge_pred: (B, 1, H, W) edge prediction  (only when return_edge=True)
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected single-channel input, got {C}"

        # ---- Step 1: Shared Patch Embedding + Norm ----
        tokens_2d = self.patch_embed(x)                          # (B, d, H/p, W/p)
        _, d, h_p, w_p = tokens_2d.shape
        tokens_seq = tokens_2d.flatten(2).transpose(1, 2)        # (B, L, d)
        tokens_seq = self.embed_norm(tokens_seq)

        # ---- Step 2: 2D Positional Encoding ----
        pos_enc = self._get_2d_sincos_pos_enc(h_p, w_p, d, x.device, x.dtype)
        tokens_seq = tokens_seq + pos_enc

        # ---- Step 3: Post-Embedding Local Mixing ----
        pre_local = self.pre_dwconv(tokens_seq.transpose(1, 2)).transpose(1, 2)
        pre_local = self.pre_norm(pre_local)
        tokens_seq = tokens_seq + self.pre_ffn(pre_local)

        # ---- Step 4a: Path 1 — Mask Mamba (bidirectional) ----
        if self.use_mask_path:
            mask_fwd = self.mask_mamba_fwd(tokens_seq)
            mask_bwd = self.mask_mamba_bwd(tokens_seq.flip(1)).flip(1)
            mask_gate = torch.sigmoid(self.mask_bidir_gate)
            mask_feat = mask_gate * mask_fwd + (1.0 - mask_gate) * mask_bwd
        else:
            mask_feat = None
        # ---- Step 4b: Path 2 — Edge Mamba (bidirectional) ----
        if self.use_edge_path:
            edge_fwd = self.edge_mamba_fwd(tokens_seq)
            edge_bwd = self.edge_mamba_bwd(tokens_seq.flip(1)).flip(1)
            edge_gate = torch.sigmoid(self.edge_bidir_gate)
            edge_feat = edge_gate * edge_fwd + (1.0 - edge_gate) * edge_bwd
        else:
            edge_feat = None
        # ---- Step 4c: Edge Head (for edge supervision) ----
        if self.use_edge_path:
            edge_2d = edge_feat.transpose(1, 2).reshape(B, d, h_p, w_p)
            edge_pred = self.edge_head(edge_2d)
            if edge_pred.shape[-2:] != x.shape[-2:]:
                edge_pred = F.interpolate(edge_pred, size=(H, W),
                                        mode='bilinear', align_corners=False)
        else:
            edge_pred = None
        # ---- Step 5: Cross-Gated Fusion ----
        if self.use_mask_path and self.use_edge_path:
            # 双路径：正常交叉门控融合
            mask_proj = self.fusion_mask_proj(mask_feat)
            edge_proj = self.fusion_edge_proj(edge_feat)
            gate = self.fusion_gate(torch.cat([mask_proj, edge_proj], dim=-1))
            fused = gate * mask_proj + (1.0 - gate) * edge_proj
            fused = self.fusion_norm(fused)
        elif self.use_mask_path:
            # 仅 Mask Mamba（消融 Edge）
            fused = self.fusion_norm(self.fusion_mask_proj(mask_feat))
        else:
            # 仅 Edge Mamba（消融 Mask）
            fused = self.fusion_norm(self.fusion_edge_proj(edge_feat))
        # ---- Step 6: Post-Mamba Local Mixing ----
        post_local = self.post_dwconv(fused.transpose(1, 2)).transpose(1, 2)
        post_local = self.post_norm(post_local)
        out_seq = fused + self.post_ffn(post_local)

        # ---- Step 7: Reshape + Unembed ----
        out_2d = out_seq.transpose(1, 2).reshape(B, d, h_p, w_p)
        delta = self.patch_unembed(out_2d)
        if delta.shape[-2:] != x.shape[-2:]:
            delta = F.interpolate(delta, size=(H, W),
                                  mode='bilinear', align_corners=False)

        # ---- Step 8: Gated Residual ----
        refined = x + self.gamma * delta

        if return_edge:
            return refined, edge_pred
        return refined

    
# ===================================================================== #
#  四向扫描序列化工具
# ===================================================================== #

def _build_scan_indices(h: int, w: int, device: torch.device):
    """
    构建四种二维扫描顺序的索引，以及对应的反索引（用于恢复原始空间位置）。

    四种扫描方向:
      1) row_fwd:   行主序正向  (0,0)->(0,1)->...→(1,0)->...  即 0,1,2,...,h*w-1
      2) row_bwd:   行主序反向  (h-1,w-1)->...→(0,0)          即 h*w-1,...,1,0
      3) col_fwd:   列主序正向  (0,0)->(1,0)->...→(0,1)->...
      4) col_bwd:   列主序反向  反向的列主序

    Returns:
      scan_indices:   (4, h*w) LongTensor  — 每行是一种扫描顺序
      unscan_indices: (4, h*w) LongTensor  — 对应反索引，用于从扫描序列恢复空间序列
    """
    L = h * w
    # 行主序 (row-major): 就是默认 flatten 顺序
    row_fwd = torch.arange(L, device=device)
    # 行主序反向
    row_bwd = row_fwd.flip(0)

    # 列主序 (column-major): 先遍历行再遍历列
    # 即 (0,0),(1,0),(2,0),...,(0,1),(1,1),...
    coords = torch.arange(L, device=device)
    row_idx = coords // w   # 行号
    col_idx = coords % w    # 列号
    # 列主序：按 (col, row) 排序
    col_fwd = (col_idx * h + row_idx)  # 这给出列主序下每个空间位置的新序号
    # 我们需要的是：列主序第 i 个位置 对应 空间中的哪个位置
    # 即 argsort(col_fwd)
    col_fwd_order = col_fwd.argsort()
    col_bwd_order = col_fwd_order.flip(0)

    scan_indices = torch.stack([row_fwd, row_bwd, col_fwd_order, col_bwd_order], dim=0)  # (4, L)

    # 反索引: unscan[scan[i]] = i  →  unscan = argsort(scan)
    unscan_indices = scan_indices.argsort(dim=1)  # (4, L)

    return scan_indices, unscan_indices


# ===================================================================== #
#  FusedScanMamba: 四向扫描 + mask/edge 混合交错 + Mamba
# ===================================================================== #

class FusedScanMamba(nn.Module):
    """
    Fused Four-Directional Scan Mamba with Mask-Edge Interleaving.

    整体流程 (以 use_mask + use_edge 为例):
    ┌─────────────────────────────────────────────────────────────┐
    │  输入: mask_tokens (B, L, d),  edge_tokens (B, L, d)       │
    │                                                             │
    │  对于每种扫描方向 k ∈ {row_fwd, row_bwd, col_fwd, col_bwd}:│
    │    1. 按 scan_indices[k] 重排 mask_tokens → mask_k          │
    │    2. 按 scan_indices[k] 重排 edge_tokens → edge_k          │
    │    3. 交错拼接: [m0, e0, m1, e1, ..., m_{L-1}, e_{L-1}]    │
    │       得到 interleaved_k: (B, 2L, d)                        │
    │    4. Mamba(interleaved_k) → out_k: (B, 2L, d)              │
    │    5. 拆分出 mask 和 edge 部分，各 (B, L, d)                │
    │    6. 用 unscan_indices[k] 恢复空间顺序                     │
    │                                                             │
    │  四方向加权融合 → 输出 mask_out, edge_out (各 B, L, d)       │
    └─────────────────────────────────────────────────────────────┘

    当仅用一种信息时 (use_mask=True, use_edge=False):
      跳过交错，直接对 mask_tokens 做四向 Mamba。

    Arguments:
      d_model:    token 维度
      n_layer:    每个方向的 Mamba 层数
      d_state:    SSM 状态维度
      use_mask:   是否使用 mask 路径
      use_edge:   是否使用 edge 路径
      attn_layer_idx: 在哪些层使用注意力替代 Mamba (空列表=纯Mamba)
      attn_cfg:   注意力层配置
    """

    def __init__(
        self,
        d_model: int = 32,
        n_layer: int = 2,
        d_state: int = 16,
        use_mask: bool = True,
        use_edge: bool = True,
        attn_layer_idx: list = None,
        attn_cfg: dict = None,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.use_edge = use_edge
        self.n_directions = 4  # row_fwd, row_bwd, col_fwd, col_bwd

        if attn_layer_idx is None:
            attn_layer_idx = []
        if attn_cfg is None:
            attn_cfg = {}

        # ---- 四个方向各一个 MixerModel（权重不共享，各自学习方向特性） ----
        self.direction_mambas = nn.ModuleList()
        for _ in range(self.n_directions):
            cfg = MambaRefineConfig(
                d_model=d_model, d_intermediate=0, n_layer=n_layer,
                ssm_cfg={"layer": "Mamba1", "d_state": d_state},
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
            )
            self.direction_mambas.append(MixerModel(**cfg.__dict__))

        # ---- 四方向加权融合 ----
        # 可学习的 per-direction, per-channel 权重
        self.direction_weights = nn.Parameter(torch.ones(self.n_directions, 1, 1, d_model) / self.n_directions)

        # ---- 扫描索引缓存 ----
        self._scan_cache = {}

    def _get_scan_indices(self, h: int, w: int, device: torch.device):
        """带缓存的获取扫描索引"""
        cache_key = (h, w, device)
        if cache_key not in self._scan_cache:
            scan_idx, unscan_idx = _build_scan_indices(h, w, device)
            self._scan_cache[cache_key] = (scan_idx, unscan_idx)
        return self._scan_cache[cache_key]

    def forward(self, mask_tokens, edge_tokens, h_p: int, w_p: int):
        """
        Args:
          mask_tokens: (B, L, d) or None  — mask 路径 token
          edge_tokens: (B, L, d) or None  — edge 路径 token
          h_p, w_p: patch grid 的高和宽 (L = h_p * w_p)

        Returns:
          mask_out: (B, L, d) or None  — 融合后的 mask 特征
          edge_out: (B, L, d) or None  — 融合后的 edge 特征
        """
        B = mask_tokens.shape[0] if mask_tokens is not None else edge_tokens.shape[0]
        d = mask_tokens.shape[2] if mask_tokens is not None else edge_tokens.shape[2]
        L = h_p * w_p
        device = mask_tokens.device if mask_tokens is not None else edge_tokens.device

        scan_indices, unscan_indices = self._get_scan_indices(h_p, w_p, device)
        # scan_indices: (4, L), unscan_indices: (4, L)

        both_paths = self.use_mask and self.use_edge and mask_tokens is not None and edge_tokens is not None

        # 收集四个方向的输出
        mask_outs = []
        edge_outs = []

        for k in range(self.n_directions):
            s_idx = scan_indices[k]    # (L,)
            us_idx = unscan_indices[k]  # (L,)

            if both_paths:
                # --- 双路径：交错混合后送入 Mamba ---
                # 按扫描顺序重排
                mask_k = mask_tokens[:, s_idx, :]   # (B, L, d)
                edge_k = edge_tokens[:, s_idx, :]   # (B, L, d)

                # 交错: [m0, e0, m1, e1, ...] → (B, 2L, d)
                interleaved = torch.stack([mask_k, edge_k], dim=2)  # (B, L, 2, d)
                interleaved = interleaved.reshape(B, 2 * L, d)

                # Mamba 处理
                out_k = self.direction_mambas[k](interleaved)  # (B, 2L, d)

                # 拆分回 mask 和 edge
                out_k = out_k.reshape(B, L, 2, d)
                mask_k_out = out_k[:, :, 0, :]  # (B, L, d)
                edge_k_out = out_k[:, :, 1, :]  # (B, L, d)

                # 恢复空间顺序
                mask_k_out = mask_k_out[:, us_idx, :]
                edge_k_out = edge_k_out[:, us_idx, :]

                mask_outs.append(mask_k_out)
                edge_outs.append(edge_k_out)

            elif self.use_mask and mask_tokens is not None:
                # --- 仅 mask 路径 ---
                mask_k = mask_tokens[:, s_idx, :]
                out_k = self.direction_mambas[k](mask_k)
                mask_k_out = out_k[:, us_idx, :]
                mask_outs.append(mask_k_out)

            elif self.use_edge and edge_tokens is not None:
                # --- 仅 edge 路径 ---
                edge_k = edge_tokens[:, s_idx, :]
                out_k = self.direction_mambas[k](edge_k)
                edge_k_out = out_k[:, us_idx, :]
                edge_outs.append(edge_k_out)

        # ---- 四方向加权融合 ----
        weights = torch.softmax(self.direction_weights, dim=0)  # (4, 1, 1, d)

        mask_out = None
        edge_out = None

        if len(mask_outs) > 0:
            mask_stack = torch.stack(mask_outs, dim=0)  # (4, B, L, d)
            mask_out = (weights * mask_stack).sum(dim=0)  # (B, L, d)

        if len(edge_outs) > 0:
            edge_stack = torch.stack(edge_outs, dim=0)
            edge_out = (weights * edge_stack).sum(dim=0)

        return mask_out, edge_out


# ===================================================================== #
#  主模型: FusedDualScanMamba
# ===================================================================== #

class FusedDualScanMamba(nn.Module):
    """
    Fused Dual-Scan Mamba for mask refinement with edge auxiliary supervision.

    ================== 整体架构 ==================

    输入: mask logits (B, 1, H, W)
    ┌──────────────────────────────────────────────────────────────────┐
    │  1. 共享 Patch Embedding: Conv2d → (B, d, h_p, w_p)             │
    │                                                                  │
    │  2. 2D 局部上下文混合 (序列化之前，在 2D 特征图上):              │
    │     - 2D DepthwiseConv (3×3) + Norm + FFN                        │
    │     → mask_2d, edge_2d: (B, d, h_p, w_p)                        │
    │                                                                  │
    │  3. Flatten → (B, L, d) + 2D Sincos PE                          │
    │     → mask_tokens, edge_tokens                                   │
    │                                                                  │
    │  4. 四向扫描 + 交错混合 Mamba (FusedScanMamba):                  │
    │     对每个方向 k:                                                │
    │       scan → interleave(mask, edge) → Mamba → deinterleave       │
    │     四方向加权融合                                                │
    │     → mask_feat, edge_feat: (B, L, d)                            │
    │                                                                  │
    │  5. Edge 辅助监督头:                                             │
    │     edge_feat → ConvHead → edge_pred (B, 1, H, W)               │
    │     训练时计算 edge loss，推理时可忽略                            │
    │                                                                  │
    │  6. 轻量校正融合 (前融合强→后融合轻):                            │
    │     gate = σ(Linear(mask_feat))                                  │
    │     fused = mask_feat + gate ⊙ edge_feat                         │
    │                                                                  │
    │  7. Reshape → (B, d, h_p, w_p) 恢复 2D 空间结构                 │
    │                                                                  │
    │  8. Post Local Mixing (2D DWConv + FFN, 与前端对称)              │
    │                                                                  │
    │  9. Patch Unembed → delta (B, 1, H, W)                          │
    │                                                                  │
    │ 10. Gated Residual: output = input + γ · delta                   │
    └──────────────────────────────────────────────────────────────────┘

    消融开关:
      use_mask_path: 关闭则无 mask token（仅 edge）
      use_edge_path: 关闭则无 edge token（仅 mask，无 edge 监督）
      attn_layer_idx: 非空则在指定层用 MHA 替代 Mamba

    References:
      - VMamba (Liu et al., ICLR 2024): Multi-direction scan + SSM
      - Vision Mamba (Zhu et al., 2024): Bidirectional SSM
      - PlainMamba (Yang et al., 2024): Four-direction scanning
    """

    def __init__(
        self,
        d_model: int = 32,
        n_layer: int = 2,
        patch_size: int = 4,
        d_state: int = 16,
        dwconv_kernel: int = 3,
        dropout: float = 0.0,
        edge_n_layer: int = 1,
        use_mask_path: bool = True,
        use_edge_path: bool = True,
        # ---- 注意力机制开关 ----
        # 传入列表指定哪些层用注意力，例如 [1] 表示第1层用 MHA
        # 空列表 [] 或 None 表示纯 Mamba
        attn_layer_idx: list = None,
        attn_cfg: dict = None,
    ):
        super().__init__()
        assert use_mask_path or use_edge_path, "At least one path must be enabled"
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4 for 2D sincos PE"

        self.use_mask_path = use_mask_path
        self.use_edge_path = use_edge_path
        self.patch_size = patch_size
        self.d_model = d_model

        if attn_layer_idx is None:
            attn_layer_idx = []
        if attn_cfg is None:
            attn_cfg = {}

        # ==================== 1. 共享 Patch Embedding ====================
        self.patch_embed = nn.Conv2d(
            1, d_model, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.embed_norm = nn.LayerNorm(d_model)

        # ==================== 2. 2D 局部上下文混合 (序列化之前) ====================
        # 在 2D 特征图上做 DepthwiseConv + FFN，真正的 2D 邻域混合
        assert dwconv_kernel % 2 == 1, "dwconv_kernel must be odd"

        # Mask 路径的 2D 局部混合
        if use_mask_path:
            self.mask_local_2d = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=dwconv_kernel,
                          padding=dwconv_kernel // 2, groups=d_model, bias=False),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=True),  # pointwise
                nn.GELU(),
            )

        # Edge 路径的 2D 局部混合
        if use_edge_path:
            self.edge_local_2d = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=dwconv_kernel,
                          padding=dwconv_kernel // 2, groups=d_model, bias=False),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=True),
                nn.GELU(),
            )

        # ==================== 3. 2D Positional Encoding ====================
        self._pos_enc_cache = {}

        # ==================== 4. 四向扫描交错 Mamba ====================
        self.fused_scan_mamba = FusedScanMamba(
            d_model=d_model,
            n_layer=n_layer,
            d_state=d_state,
            use_mask=use_mask_path,
            use_edge=use_edge_path,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
        )

        # ==================== 5. Edge 辅助监督头 ====================
        # 训练时用 GT edge 监督，推理时可忽略输出
        if use_edge_path:
            self.edge_head = nn.Sequential(
                nn.ConvTranspose2d(d_model, d_model // 2,
                                   kernel_size=patch_size, stride=patch_size, bias=True),
                nn.BatchNorm2d(d_model // 2),
                nn.GELU(),
                nn.Conv2d(d_model // 2, d_model // 4, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(d_model // 4),
                nn.GELU(),
                nn.Conv2d(d_model // 4, 1, kernel_size=1, bias=True),
            )

        # ==================== 6. 轻量校正融合 ====================
        # 前融合强（Mamba内交错）→ 后融合轻（残差门控校正）
        if use_mask_path and use_edge_path:
            # 轻量门控: gate = σ(W · mask_feat), fused = mask + gate ⊙ edge
            self.correction_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            self.correction_norm = nn.LayerNorm(d_model)

        # ==================== 7. Post Local Mixing (2D，与前端对称) ====================
        # 在 reshape 回 2D 之后做真正的空间邻域混合，而不是在 1D 序列上
        self.post_local_2d = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3,
                      padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=True),  # pointwise FFN
            nn.GELU(),
        )

        # ==================== 8. Patch Unembedding ====================
        self.patch_unembed = nn.ConvTranspose2d(
            d_model, 1, kernel_size=patch_size, stride=patch_size, bias=True
        )

        # ==================== 9. Gated Residual ====================
        self.gamma = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #
    #  2D Sincos Positional Encoding
    # ------------------------------------------------------------------ #
    def _get_2d_sincos_pos_enc(self, h, w, d_model, device, dtype):
        cache_key = (h, w, d_model, device)
        if cache_key in self._pos_enc_cache:
            return self._pos_enc_cache[cache_key].to(dtype=dtype)

        half_d = d_model // 2
        row_pos = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
        col_pos = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1)
        dim_idx = torch.arange(0, half_d, 2, device=device, dtype=torch.float32)
        freq = 1.0 / (10000.0 ** (dim_idx / half_d))

        row_enc = torch.cat([torch.sin(row_pos * freq), torch.cos(row_pos * freq)], dim=-1)
        col_enc = torch.cat([torch.sin(col_pos * freq), torch.cos(col_pos * freq)], dim=-1)

        row_enc = row_enc.unsqueeze(1).expand(-1, w, -1)
        col_enc = col_enc.unsqueeze(0).expand(h, -1, -1)
        pos_enc = torch.cat([row_enc, col_enc], dim=-1).reshape(1, h * w, d_model)

        self._pos_enc_cache[cache_key] = pos_enc
        return pos_enc.to(dtype=dtype)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, x, return_edge=False):
        """
        Args:
            x: (B, 1, H, W) single-channel mask logits
            return_edge: 是否返回 edge 预测（训练时为 True 用于辅助监督）

        Returns:
            refined: (B, 1, H, W) refined mask logits
            edge_pred: (B, 1, H, W) or None — edge 预测（仅 return_edge=True 且 use_edge_path=True）
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected single-channel input, got {C}"

        # ---- Step 1: 共享 Patch Embedding ----
        tokens_2d = self.patch_embed(x)              # (B, d, h_p, w_p)
        _, d, h_p, w_p = tokens_2d.shape

        # ---- Step 2: 2D 局部上下文混合 (序列化之前) ----
        # 在真正的 2D 特征图上做邻域混合，比 1D DWConv 更符合图像先验
        if self.use_mask_path:
            mask_2d = tokens_2d + self.mask_local_2d(tokens_2d)   # 残差连接
        else:
            mask_2d = None

        if self.use_edge_path:
            edge_2d = tokens_2d + self.edge_local_2d(tokens_2d)
        else:
            edge_2d = None

        # ---- Step 3: Flatten + 2D Positional Encoding ----
        pos_enc = self._get_2d_sincos_pos_enc(h_p, w_p, d, x.device, x.dtype)

        mask_tokens = None
        edge_tokens = None

        if mask_2d is not None:
            mask_tokens = mask_2d.flatten(2).transpose(1, 2)      # (B, L, d)
            mask_tokens = self.embed_norm(mask_tokens)
            mask_tokens = mask_tokens + pos_enc

        if edge_2d is not None:
            edge_tokens = edge_2d.flatten(2).transpose(1, 2)
            edge_tokens = self.embed_norm(edge_tokens)
            edge_tokens = edge_tokens + pos_enc

        # ---- Step 4: 四向扫描 + 交错混合 Mamba ----
        mask_feat, edge_feat = self.fused_scan_mamba(
            mask_tokens, edge_tokens, h_p, w_p
        )

        # ---- Step 5: Edge 辅助监督头 ----
        edge_pred = None
        if self.use_edge_path and edge_feat is not None:
            edge_2d_feat = edge_feat.transpose(1, 2).reshape(B, d, h_p, w_p)
            edge_pred = self.edge_head(edge_2d_feat)
            if edge_pred.shape[-2:] != x.shape[-2:]:
                edge_pred = F.interpolate(edge_pred, size=(H, W),
                                          mode='bilinear', align_corners=False)

        # ---- Step 6: 轻量校正融合 ----
        if self.use_mask_path and self.use_edge_path and mask_feat is not None and edge_feat is not None:
            # 前端已在 Mamba 内部交错混合，后端只做轻量残差校正
            gate = self.correction_gate(mask_feat)        # (B, L, d) → σ
            fused = mask_feat + gate * edge_feat           # 残差式校正
            fused = self.correction_norm(fused)
        elif mask_feat is not None:
            fused = mask_feat
        else:
            fused = edge_feat

        # ---- Step 7: Reshape to 2D ----
        out_2d = fused.transpose(1, 2).reshape(B, d, h_p, w_p)

        # ---- Step 8: Post Local Mixing (2D, 与前端对称) ----
        # reshape 回 2D 后做真正的 2D 空间邻域混合
        out_2d = out_2d + self.post_local_2d(out_2d)   # 残差连接

        # ---- Step 9: Patch Unembed ----
        delta = self.patch_unembed(out_2d)
        if delta.shape[-2:] != x.shape[-2:]:
            delta = F.interpolate(delta, size=(H, W), mode='bilinear', align_corners=False)

        # ---- Step 10: Gated Residual ----
        refined = x + self.gamma * delta

        if return_edge:
            return refined, edge_pred
        return refined




if __name__ == "__main__":
    model = MambaConv().to('cuda')
    x = torch.randn(1, 128, 64).to('cuda')
    print(model(x).shape)
