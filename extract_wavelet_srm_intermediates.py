import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_uint8_gray(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        out = np.zeros_like(arr, dtype=np.uint8)
    else:
        out = ((arr - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
    return out


def _to_uint8_gray_percentile(x: torch.Tensor, percentile: float = 99.0) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    lo = float(np.percentile(arr, 100.0 - percentile))
    hi = float(np.percentile(arr, percentile))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    out = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return out


def _to_uint8_signed(x: torch.Tensor, percentile: float = 99.0) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    vmax = float(np.percentile(np.abs(arr), percentile))
    if vmax < 1e-12:
        return np.full_like(arr, 127, dtype=np.uint8)
    out = ((arr / vmax) * 127.0 + 127.0).clip(0, 255).astype(np.uint8)
    return out


def _to_uint8_rgb(x_chw: torch.Tensor) -> np.ndarray:
    ch = []
    for c in range(x_chw.shape[0]):
        ch.append(_to_uint8_gray(x_chw[c]))
    return np.stack(ch, axis=-1)


def _save_gray(x: torch.Tensor, out_path: Path) -> None:
    img = _to_uint8_gray(x)
    cv2.imwrite(str(out_path), img)


def _save_gray_percentile(x: torch.Tensor, out_path: Path, percentile: float = 99.0) -> None:
    img = _to_uint8_gray_percentile(x, percentile=percentile)
    cv2.imwrite(str(out_path), img)


def _save_signed_gray(x: torch.Tensor, out_path: Path, percentile: float = 99.0) -> None:
    img = _to_uint8_signed(x, percentile=percentile)
    cv2.imwrite(str(out_path), img)


def _save_signed_heatmap(x: torch.Tensor, out_path: Path, percentile: float = 99.0) -> None:
    img = _to_uint8_signed(x, percentile=percentile)
    # TURBO colormap gives stronger local contrast than plain grayscale.
    vis = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_path), vis)


def _save_rgb(x_chw: torch.Tensor, out_path: Path) -> None:
    img_rgb = _to_uint8_rgb(x_chw)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), img_bgr)


def _haar_kernels(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    h0 = torch.tensor([1.0, 1.0], dtype=dtype, device=device) / np.sqrt(2.0)
    h1 = torch.tensor([-1.0, 1.0], dtype=dtype, device=device) / np.sqrt(2.0)
    ll = torch.outer(h0, h0)
    lh = torch.outer(h0, h1)
    hl = torch.outer(h1, h0)
    hh = torch.outer(h1, h1)
    return torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)


def _srm_kernels(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    k1 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, -0.25, 0.0],
            [0.0, 0.5, -1.0, 0.5, 0.0],
            [0.0, -0.25, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    k2 = torch.tensor(
        [
            [-1.0 / 12.0, 2.0 / 12.0, -2.0 / 12.0, 2.0 / 12.0, -1.0 / 12.0],
            [2.0 / 12.0, -6.0 / 12.0, 8.0 / 12.0, -6.0 / 12.0, 2.0 / 12.0],
            [-2.0 / 12.0, 8.0 / 12.0, -1.0, 8.0 / 12.0, -2.0 / 12.0],
            [2.0 / 12.0, -6.0 / 12.0, 8.0 / 12.0, -6.0 / 12.0, 2.0 / 12.0],
            [-1.0 / 12.0, 2.0 / 12.0, -2.0 / 12.0, 2.0 / 12.0, -1.0 / 12.0],
        ],
        dtype=dtype,
        device=device,
    )
    k3 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, -1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    return torch.stack([k1, k2, k3], dim=0).unsqueeze(1)


def _load_image_as_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return x


def _save_response_family(
    x: torch.Tensor,
    base_path: Path,
    percentile: float,
    signed: bool = True,
) -> None:
    _save_gray(x, base_path.with_suffix(".png"))
    _save_gray(torch.abs(x), base_path.with_name(base_path.name + "_abs").with_suffix(".png"))
    _save_gray_percentile(
        x,
        base_path.with_name(base_path.name + "_pctl").with_suffix(".png"),
        percentile=percentile,
    )
    if signed:
        _save_signed_gray(
            x,
            base_path.with_name(base_path.name + "_signed").with_suffix(".png"),
            percentile=percentile,
        )
        _save_signed_heatmap(
            x,
            base_path.with_name(base_path.name + "_signed_heat").with_suffix(".png"),
            percentile=percentile,
        )


def run(
    image_path: Path,
    output_dir: Path,
    srm_threshold: float,
    save_npy: bool,
    viz_percentile: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_wavelet = output_dir / "wavelet"
    out_srm = output_dir / "srm"
    out_fusion = output_dir / "fusion"
    _ensure_dir(out_wavelet)
    _ensure_dir(out_srm)
    _ensure_dir(out_fusion)

    x = _load_image_as_tensor(image_path, device=device)  # [1,3,H,W]
    _, c, h, w = x.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {c}")

    channel_names = ["R", "G", "B"]

    # 1) RGB three channels: IR, IG, IB
    for i, name in enumerate(channel_names):
        _save_response_family(x[0, i], out_wavelet / f"I{name}", percentile=viz_percentile, signed=False)
        _save_response_family(x[0, i], out_srm / f"I{name}", percentile=viz_percentile, signed=False)

    # 2) Upsampled RGB channels: IR2, IG2, IB2
    x_up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    for i, name in enumerate(channel_names):
        _save_response_family(x_up[0, i], out_wavelet / f"I{name}2", percentile=viz_percentile, signed=False)
        _save_response_family(x_up[0, i], out_srm / f"I{name}2", percentile=viz_percentile, signed=False)

    # ---------------- Wavelet middle results ----------------
    haar = _haar_kernels(x_up.dtype, x_up.device)  # [4,1,2,2]
    w_haar = haar.repeat(3, 1, 1, 1)  # [12,1,2,2]
    coeffs = F.conv2d(x_up, w_haar, stride=2, groups=3).view(1, 3, 4, h, w)

    wavelet_hf = []
    for i, name in enumerate(channel_names):
        ll = coeffs[0, i, 0]
        lh = coeffs[0, i, 1]
        hl = coeffs[0, i, 2]
        hh = coeffs[0, i, 3]

        _save_response_family(ll, out_wavelet / f"Ill_{name}", percentile=viz_percentile, signed=True)
        _save_response_family(hl, out_wavelet / f"Ihl_{name}", percentile=viz_percentile, signed=True)
        _save_response_family(lh, out_wavelet / f"Ilh_{name}", percentile=viz_percentile, signed=True)
        _save_response_family(hh, out_wavelet / f"Ihh_{name}", percentile=viz_percentile, signed=True)

        hf = lh.abs() + hl.abs() + hh.abs()
        wavelet_hf.append(hf)
        _save_response_family(hf, out_wavelet / f"IHF_{name}", percentile=viz_percentile, signed=False)

    wavelet_features = torch.stack(wavelet_hf, dim=0)  # [3,H,W]
    _save_rgb(wavelet_features, out_wavelet / "wavelet_features.png")

    # ---------------- SRM middle results ----------------
    srm_k = _srm_kernels(x.dtype, x.device)  # [3,1,5,5]
    w_srm = srm_k.repeat(3, 1, 1, 1)  # [9,1,5,5]
    srm_resp = F.conv2d(x, w_srm, padding=2, groups=3).view(1, 3, 3, h, w)
    srm_resp = torch.clamp(srm_resp, -srm_threshold, srm_threshold)

    srm_hf = []
    for i, name in enumerate(channel_names):
        r1 = srm_resp[0, i, 0]
        r2 = srm_resp[0, i, 1]
        r3 = srm_resp[0, i, 2]

        _save_response_family(r1, out_srm / f"Isrm1_{name}", percentile=viz_percentile, signed=True)
        _save_response_family(r2, out_srm / f"Isrm2_{name}", percentile=viz_percentile, signed=True)
        _save_response_family(r3, out_srm / f"Isrm3_{name}", percentile=viz_percentile, signed=True)

        hf = r1.abs() + r2.abs() + r3.abs()
        srm_hf.append(hf)
        _save_response_family(hf, out_srm / f"IHF_{name}", percentile=viz_percentile, signed=False)

    srm_features = torch.stack(srm_hf, dim=0)  # [3,H,W]
    _save_rgb(srm_features, out_srm / "srm_features.png")

    # ---------------- Fusion features ----------------
    fw = F.instance_norm(wavelet_features.unsqueeze(0))  # [1,3,H,W]
    fs = F.instance_norm(srm_features.unsqueeze(0))      # [1,3,H,W]

    fusion_concat = torch.cat([fw, fs], dim=1)  # [1,6,H,W]

    # Deterministic gate: energy-based per-pixel weighting
    ew = fw.abs().mean(dim=1, keepdim=True)
    es = fs.abs().mean(dim=1, keepdim=True)
    gate = ew / (ew + es + 1e-6)
    fused_features = gate * fw + (1.0 - gate) * fs  # [1,3,H,W]

    # Visualize concatenation as left(wavelet) + right(srm)
    left = _to_uint8_rgb(fw[0])
    right = _to_uint8_rgb(fs[0])
    concat_vis = np.concatenate([left, right], axis=1)
    concat_vis_bgr = cv2.cvtColor(concat_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_fusion / "fusion_concat6_vis.png"), concat_vis_bgr)

    _save_gray(gate[0, 0], out_fusion / "fusion_gate.png")
    _save_gray_percentile(gate[0, 0], out_fusion / "fusion_gate_pctl.png", percentile=viz_percentile)
    _save_rgb(fused_features[0], out_fusion / "fused_features.png")

    if save_npy:
        np.save(out_wavelet / "wavelet_coeffs.npy", coeffs[0].detach().cpu().numpy())
        np.save(out_wavelet / "wavelet_features.npy", wavelet_features.detach().cpu().numpy())
        np.save(out_srm / "srm_responses.npy", srm_resp[0].detach().cpu().numpy())
        np.save(out_srm / "srm_features.npy", srm_features.detach().cpu().numpy())
        np.save(out_fusion / "fusion_concat6.npy", fusion_concat[0].detach().cpu().numpy())
        np.save(out_fusion / "fused_features.npy", fused_features[0].detach().cpu().numpy())

    print("Done.")
    print(f"Input: {image_path}")
    print(f"Output root: {output_dir}")
    print(f"Wavelet features shape: {tuple(wavelet_features.shape)}")
    print(f"SRM features shape: {tuple(srm_features.shape)}")
    print(f"Fusion concat shape: {tuple(fusion_concat.shape)}")
    print(f"Fused features shape: {tuple(fused_features.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all wavelet/SRM intermediate results and fusion features for one RGB image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/wavelet_srm_debug",
        help="Directory to save intermediate feature maps",
    )
    parser.add_argument(
        "--srm-threshold",
        type=float,
        default=3.0,
        help="Clamp threshold T for SRM responses",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save major tensors as .npy files",
    )
    parser.add_argument(
        "--viz-percentile",
        type=float,
        default=99.0,
        help="Percentile used in robust visualization (recommended: 98-99.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        srm_threshold=args.srm_threshold,
        save_npy=args.save_npy,
        viz_percentile=args.viz_percentile,
    )


if __name__ == "__main__":
    main()
