# src/metrics_fidelity.py
"""Fidelity Metrics for Explanation Quality Assessment

This module implements fidelity metrics that evaluate explanation quality by
measuring how much a model's prediction changes when important regions (according
to the explanation) are masked or revealed. Metrics include Average Drop, Average
Gain, Insertion AUC, and Deletion AUC.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


__all__ = [
    "average_drop_increase",
    "average_gain",
    "insertion_auc",
    "deletion_auc",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _normalize_heatmap(hm: torch.Tensor) -> torch.Tensor:
    """Normalize heatmap to shape (1,1,H,W) with values in [0,1]."""
    if hm.dim() == 2:
        hm = hm.unsqueeze(0).unsqueeze(0)
    hm = hm.to(DEVICE)
    hm = hm - hm.amin(dim=(-2, -1), keepdim=True)
    hm = hm / (hm.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return hm

def _upsample_heatmap_to_image(
    hm: torch.Tensor, H: int, W: int, patchwise: bool = True
) -> torch.Tensor:
    """Upsample heatmap to image resolution."""
    mode = "nearest" if patchwise else "bilinear"
    return F.interpolate(
        hm,
        size=(H, W),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    ).to(DEVICE)

def _to_pixel_space(
    x_norm: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...], inverse: bool = True
) -> torch.Tensor:
    """Convert image between normalized and pixel space."""
    mean = torch.tensor(mean, device=DEVICE)[None, :, None, None]
    std = torch.tensor(std, device=DEVICE)[None, :, None, None]
    return x_norm * std + mean if inverse else (x_norm - mean) / std

def _build_blur_baseline(
    x_norm: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    ksize: int = 21,
    sigma: float = 7.0,
) -> torch.Tensor:
    """Create a blurred baseline image for masking."""
    x_pix = _to_pixel_space(x_norm, mean, std, inverse=True).clamp(0, 1)
    B_pix = gaussian_blur(x_pix, kernel_size=ksize, sigma=sigma)
    return _to_pixel_space(B_pix, mean, std, inverse=False)

def _compose_masked(
    x_norm: torch.Tensor, mask01: torch.Tensor, baseline_norm: torch.Tensor
) -> torch.Tensor:
    """Blend original and baseline using mask."""
    mask01 = mask01.to(DEVICE)
    x_norm = x_norm.to(DEVICE)
    baseline_norm = baseline_norm.to(DEVICE)
    return mask01 * x_norm + (1 - mask01) * baseline_norm

def _trapezoid_auc(xs: torch.Tensor, ys: torch.Tensor) -> float:
    """Compute area under curve using trapezoidal rule."""
    return torch.trapz(ys.to(DEVICE), xs.to(DEVICE)).item()

@torch.no_grad()
def average_drop_increase(
    model,
    x_norm: torch.Tensor,
    class_idx: int,
    heatmap_2d: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    keep_ratio: float = 0.25,
    patchwise: bool = True,
    blur_ksize: int = 21,
    blur_sigma: float = 7.0,
) -> Tuple[float, int, float, float]:
    """Compute Average Drop and increase metrics.

    Average Drop measures the relative decrease in logit when keeping only
    top-k important regions (according to the heatmap).

    Returns
    -------
    Tuple[float, int, float, float]
        (average_drop_pct, increase_flag, full_logit, masked_logit)
    """
    model = model.to(DEVICE).eval()
    x_norm = x_norm.to(DEVICE)

    s = model(x_norm).logits[0, class_idx].item()

    _, _, H, W = x_norm.shape
    hm = _normalize_heatmap(heatmap_2d)
    hm_im = _upsample_heatmap_to_image(hm, H, W, patchwise=patchwise).squeeze()
    thresh = torch.quantile(hm_im.flatten(), 1 - keep_ratio)
    mask01 = (hm_im >= thresh).float()[None, None, ...].to(DEVICE)

    baseline_norm = _build_blur_baseline(x_norm, mean, std, ksize=blur_ksize, sigma=blur_sigma)
    x_masked = _compose_masked(x_norm, mask01, baseline_norm)

    s_m = model(x_masked).logits[0, class_idx].item()

    drop = max(0.0, (s - s_m)) / (abs(s) + 1e-8) * 100.0
    increase = 1 if s_m > s else 0

    return float(drop), int(increase), float(s), float(s_m)
    
@torch.no_grad()
def average_gain(
    model,
    x_norm: torch.Tensor,
    class_idx: int,
    heatmap_2d: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    keep_ratio: float = 0.25,
    patchwise: bool = True,
    blur_ksize: int = 21,
    blur_sigma: float = 7.0,
) -> Tuple[float, float, float]:
    """Compute Average Gain metric.

    Average Gain measures the relative increase in class probability when
    keeping only top-k important regions:

        AG = [o_c - p_c]⁺ / (1 - p_c) × 100

    where p_c is the probability on the full image and o_c is on the masked image.

    Returns
    -------
    Tuple[float, float, float]
        (average_gain, prob_full, prob_masked)
    """
    model = model.to(DEVICE).eval()
    x_norm = x_norm.to(DEVICE)

    logits_full = model(x_norm).logits[0]
    p_full = F.softmax(logits_full, dim=0)[class_idx].item()

    _, _, H, W = x_norm.shape
    hm = _normalize_heatmap(heatmap_2d)
    hm_im = _upsample_heatmap_to_image(hm, H, W, patchwise=patchwise).squeeze()

    thresh = torch.quantile(hm_im.flatten(), 1 - keep_ratio)
    mask01 = (hm_im >= thresh).float()[None, None, ...].to(DEVICE)

    baseline_norm = _build_blur_baseline(x_norm, mean, std, blur_ksize, blur_sigma)
    x_masked = _compose_masked(x_norm, mask01, baseline_norm)

    logits_masked = model(x_masked).logits[0]
    p_masked = F.softmax(logits_masked, dim=0)[class_idx].item()

    numerator = max(0.0, p_masked - p_full)
    denom = max(1e-8, 1.0 - p_full)
    ag = (numerator / denom) * 100.0

    return float(ag), float(p_full), float(p_masked)

@torch.no_grad()
def insertion_auc(
    model,
    x_norm: torch.Tensor,
    class_idx: int,
    heatmap_2d: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    steps: int = 100,
    patchwise: bool = True,
    blur_ksize: int = 21,
    blur_sigma: float = 7.0,
) -> float:
    """Compute Insertion AUC metric.

    Insertion AUC measures explanation quality by gradually revealing pixels
    (from high to low heatmap values) and measuring the area under the curve
    of the model's confidence as more pixels are revealed.

    Returns
    -------
    float
        Insertion AUC score.
    """
    model = model.to(DEVICE).eval()
    x_norm = x_norm.to(DEVICE)
    _, C, H, W = x_norm.shape

    hm = _normalize_heatmap(heatmap_2d)
    hm_im = _upsample_heatmap_to_image(hm, H, W, patchwise=patchwise)
    imp = hm_im.view(-1)

    order = torch.argsort(imp, descending=True)
    N = order.numel()
    batch = max(1, N // steps)

    baseline_norm = _build_blur_baseline(x_norm, mean, std, blur_ksize, blur_sigma)
    canvas = baseline_norm.clone()

    full_logit = model(x_norm).logits[0, class_idx].item()
    base_logit = model(baseline_norm).logits[0, class_idx].item()

    xs, ys = [0.0], [base_logit]
    for i in range(0, N, batch):
        j = min(i + batch, N)
        idx = order[i:j]
        y = (idx // W)
        x = (idx % W)
        canvas[:, :, y, x] = x_norm[:, :, y, x]
        frac = j / N
        score = model(canvas).logits[0, class_idx].item()
        xs.append(frac)
        ys.append(score)

    xs, ys = torch.tensor(xs, device=DEVICE), torch.tensor(ys, device=DEVICE)
    denom = max(1e-8, full_logit - base_logit)
    ys_norm = ((ys - base_logit) / denom).clamp(0, 1)
    return _trapezoid_auc(xs, ys_norm)

@torch.no_grad()
def deletion_auc(
    model,
    x_norm: torch.Tensor,
    class_idx: int,
    heatmap_2d: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    steps: int = 100,
    patchwise: bool = True,
    blur_ksize: int = 21,
    blur_sigma: float = 7.0,
) -> float:
    """Compute Deletion AUC metric.

    Deletion AUC measures explanation quality by gradually removing pixels
    (from high to low heatmap values) and measuring the area under the curve
    of the model's confidence as more pixels are deleted.

    Returns
    -------
    float
        Deletion AUC score.
    """
    model = model.to(DEVICE).eval()
    x_norm = x_norm.to(DEVICE)
    _, C, H, W = x_norm.shape

    hm = _normalize_heatmap(heatmap_2d)
    hm_im = _upsample_heatmap_to_image(hm, H, W, patchwise=patchwise)
    imp = hm_im.view(-1)

    order = torch.argsort(imp, descending=True)
    N = order.numel()
    batch = max(1, N // steps)

    baseline_norm = _build_blur_baseline(x_norm, mean, std, blur_ksize, blur_sigma)
    canvas = x_norm.clone()

    full_logit = model(x_norm).logits[0, class_idx].item()
    base_logit = model(baseline_norm).logits[0, class_idx].item()

    xs, ys = [0.0], [full_logit]
    for i in range(0, N, batch):
        j = min(i + batch, N)
        idx = order[i:j]
        y = (idx // W)
        x = (idx % W)
        canvas[:, :, y, x] = baseline_norm[:, :, y, x]
        frac = j / N
        score = model(canvas).logits[0, class_idx].item()
        xs.append(frac)
        ys.append(score)

    xs, ys = torch.tensor(xs, device=DEVICE), torch.tensor(ys, device=DEVICE)
    denom = max(1e-8, full_logit - base_logit)
    ys_norm = ((ys - base_logit) / denom).clamp(0, 1)
    return _trapezoid_auc(xs, ys_norm)
