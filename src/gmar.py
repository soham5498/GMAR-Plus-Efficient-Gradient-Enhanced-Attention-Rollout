"""GMAR: Gradient-weighted Multi-head Attention Rollout

This module implements a compact, well-documented GMAR helper class used to
compute class-specific attention heatmaps for transformer-based vision models
(e.g., ViT). The public API is intentionally small and stable so it can be
imported and used by the rest of the project without changing behavior.
"""

from pathlib import Path
from typing import List, Union
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


__all__ = ["GMAR"]


class GMAR:
    """Compute and save GMAR attention heatmaps.

    Parameters
    ----------
    alpha : float, optional
        Residual scaling added during rollout (default: 1.0).
    norm_type : str, optional
        How to compute head importance from gradients; either ``'l2'`` or
        ``'l1'`` (default: 'l2').
    """

    def __init__(self, alpha: float = 1.0, norm_type: str = "l2") -> None:
        if norm_type not in ("l1", "l2"):
            raise ValueError("norm_type must be 'l1' or 'l2'")
        self.alpha = float(alpha)
        self.norm_type = norm_type

    def compute(
        self,
        logits: torch.Tensor,
        pred_class: int,
        attn_weights: List[torch.Tensor],
        model,
    ) -> torch.Tensor:
        """Compute a normalized GMAR heatmap.

        Parameters
        ----------
        logits
            Model output logits for the current sample (shape [1, num_classes]).
        pred_class
            Index of the target class used to backprop and weight attention.
        attn_weights
            Per-layer attention tensors. Each tensor must have gradients
            (call ``retain_grad()`` on these tensors during the forward pass).
        model
            The model instance; only used to call ``zero_grad()`` before
            backward.

        Returns
        -------
        torch.Tensor
            2D heatmap of shape (S, S) with values normalized to [0, 1].
        """
        if not attn_weights:
            raise ValueError("attn_weights must be a non-empty list")

        device = attn_weights[0].device
        model.zero_grad()

        # Backprop on the chosen class to populate gradients on attention
        target_logit = logits[0, pred_class]
        target_logit.backward(retain_graph=True)

        weighted_attns: List[torch.Tensor] = []
        eps = 1e-12

        for attn in attn_weights:
            grad = attn.grad
            if grad is None:
                raise RuntimeError(
                    "attn.grad is None — ensure `.retain_grad()` was called on attention tensors"
                )

            if self.norm_type == "l2":
                head_importance = grad.pow(2).sum(dim=(-1, -2)).sqrt()[0]
            else:
                head_importance = grad.abs().sum(dim=(-1, -2))[0]

            denom = head_importance.sum().clamp_min(eps)
            head_weights = (head_importance / denom).view(1, -1, 1, 1)

            # Collapse head dimension and keep the [N, N] attention matrix
            A_weighted = (attn * head_weights).sum(dim=1)
            weighted_attns.append(A_weighted[0])

        # Residual-aware rollout
        N = weighted_attns[0].size(-1)
        I = torch.eye(N, device=device)
        rollout = I.clone()
        for A in weighted_attns:
            rollout = rollout @ A + self.alpha * I

        # Influence of class token on patches (exclude cls token at index 0)
        cls_influence = rollout[0, 1:]
        side = int((cls_influence.numel()) ** 0.5)
        cls_map = cls_influence.view(side, side).detach().cpu()

        # Safe normalization to [0, 1]
        mn = cls_map.min()
        mx = cls_map.max()
        if (mx - mn) <= 0:
            return torch.zeros_like(cls_map)
        cls_map = (cls_map - mn) / (mx - mn + 1e-8)
        return cls_map

    def save_overlay(
        self,
        cls_map: torch.Tensor,
        original_image: Image.Image,
        original_image_path: Union[str, Path],
    ) -> Path:
        """Save a heatmap overlay next to the original image and return path.

        The saved filename pattern is ``{original_stem}_gmar.png`` and the file
        is written into the same directory as ``original_image_path``.
        """
        out_path = Path(original_image_path)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        output_path = out_dir / f"{out_path.stem}_gmar.png"

        W, H = original_image.size
        heatmap = np.array(cls_map)

        hm_min, hm_max = heatmap.min(), heatmap.max()
        if (hm_max - hm_min) <= 0:
            heatmap_resized = np.zeros((H, W), dtype=np.uint8)
        else:
            heatmap_norm = (heatmap - hm_min) / (hm_max - hm_min + 1e-8)
            heatmap_img = Image.fromarray((heatmap_norm * 255).astype(np.uint8))
            heatmap_resized = np.array(heatmap_img.resize((W, H), resample=Image.BILINEAR))

        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(original_image)
        ax.imshow(heatmap_resized, cmap="jet", alpha=0.5)
        ax.axis("off")

        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return output_path
