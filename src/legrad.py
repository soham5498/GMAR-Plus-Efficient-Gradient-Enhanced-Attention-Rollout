
"""LeGrad: Gradient-weighted Attention Maps for ViT Explainability

This module implements LeGrad, a method for computing gradient-weighted attention
maps to explain Vision Transformer (ViT) predictions at the patch level. LeGrad
computes layer-wise attention gradients, merges them across layers, and generates
visual heatmap overlays for interpretability.
"""

from pathlib import Path
from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


__all__ = ["LeGradExplainer"]


class LeGradExplainer:
    """Compute and save LeGrad gradient-weighted attention heatmaps.

    LeGrad backpropagates from the predicted logit, applies ReLU to gradients,
    and merges patch importance scores across all transformer layers to create
    a final per-patch heatmap.

    Parameters
    ----------
    attn_weights : List[torch.Tensor]
        Attention matrices from each ViT block, each of shape [B, H, N, N].
    logits : torch.Tensor
        Output logits from the ViT classifier (shape [B, num_classes]).
    predicted_class : int
        Index of the target class for gradient computation.
    predicted_label : str, optional
        Human-readable label of the predicted class (default: '').
    """

    def __init__(
        self,
        attn_weights: List[torch.Tensor],
        logits: torch.Tensor,
        predicted_class: int,
        predicted_label: str = "",
    ) -> None:
        self.attn_weights = attn_weights
        self.logits = logits
        self.predicted_class = predicted_class
        self.predicted_label = predicted_label
        self.layer_maps: List[torch.Tensor] = []

    def compute_layer_maps(self) -> List[torch.Tensor]:
        """Compute per-layer patch importance from attention gradients.

        Backpropagates from the target class logit to obtain gradients on all
        attention matrices. For each layer, applies ReLU clamping to retain
        positive gradients, averages over heads, removes the CLS token, and
        computes patch-level importance scores.

        Returns
        -------
        List[torch.Tensor]
            Per-layer patch importance maps (excluding CLS token).
        """
        target_logit = self.logits[0, self.predicted_class]
        self.logits.grad = None
        target_logit.backward()

        layer_maps: List[torch.Tensor] = []
        for attn in self.attn_weights:
            grad = attn.grad
            if grad is None:
                raise RuntimeError(
                    "attn.grad is None — ensure `.retain_grad()` was called on attention tensors"
                )

            grad_pos = grad.clamp(min=0)
            gcam_mean = grad_pos.mean(dim=1)
            gcam_no_cls = gcam_mean[:, 1:, 1:]
            patch_score = gcam_no_cls.mean(dim=-1)

            layer_maps.append(patch_score)

        self.layer_maps = layer_maps
        return layer_maps

    def merge_heatmap(self) -> np.ndarray:
        """Merge layer-wise patch scores into a single normalized heatmap.

        Averages the patch importance scores across all layers and normalizes
        the result to [0, 1].

        Returns
        -------
        np.ndarray
            Final normalized 1D patch heatmap (length = num_patches).
        """
        if not self.layer_maps:
            raise ValueError("layer_maps is empty; call compute_layer_maps() first")

        merged = torch.stack(self.layer_maps).mean(dim=0)
        heatmap = merged.detach().squeeze().cpu().numpy()

        hm_min, hm_max = heatmap.min(), heatmap.max()
        if (hm_max - hm_min) <= 0:
            return np.zeros_like(heatmap)
        heatmap = (heatmap - hm_min) / (hm_max - hm_min + 1e-8)
        return heatmap

    def save_overlay(
        self,
        img: Image.Image,
        heatmap: Union[np.ndarray, torch.Tensor],
        original_image_path: Union[str, Path],
    ) -> Path:
        """Save the heatmap overlay next to the original image and return path.

        The saved filename pattern is ``{original_stem}_legrad.png`` and the file
        is written into the same directory as ``original_image_path``.

        Parameters
        ----------
        img
            Original input image (PIL.Image).
        heatmap
            Patch-level heatmap tensor or array (1D or 2D).
        original_image_path
            Path to the original image file.

        Returns
        -------
        Path
            Path where the overlay was saved.
        """
        out_path = Path(original_image_path)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        output_path = out_dir / f"{out_path.stem}_legrad.png"

        W, H = img.size
        img_np = np.array(img, dtype=np.float32) / 255.0

        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()

        if heatmap.ndim == 1:
            side = int(np.sqrt(heatmap.shape[0]))
            if side * side != heatmap.shape[0]:
                raise ValueError(
                    f"Heatmap length {heatmap.shape[0]} is not a perfect square"
                )
            heatmap_grid = heatmap.reshape(side, side)
        elif heatmap.ndim == 2:
            heatmap_grid = heatmap
        else:
            raise ValueError(f"Heatmap must be 1D or 2D, got shape {heatmap.shape}")

        heatmap_tensor = torch.tensor(heatmap_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heatmap_upsampled = F.interpolate(
            heatmap_tensor, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        hm_min, hm_max = float(heatmap_upsampled.min()), float(heatmap_upsampled.max())
        if (hm_max - hm_min) > 0:
            heatmap_upsampled = (heatmap_upsampled - hm_min) / (hm_max - hm_min + 1e-8)

        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(img_np)
        ax.imshow(heatmap_upsampled, cmap="jet", alpha=0.5)
        ax.axis("off")

        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return output_path
