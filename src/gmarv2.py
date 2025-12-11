"""GMARv2: Enhanced Gradient-weighted Multi-head Attention Rollout

This module implements an improved variant of GMAR that uses ReLU-clamped
gradients and row-normalized residual rollout for more robust attention-based
explanations in transformer vision models (e.g., ViT).
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


__all__ = ["GMARv2"]


class GMARv2:
    """Compute and save GMARv2 attention heatmaps with ReLU-clamped gradients.

    This variant improves upon standard GMAR by:
    - Clamping gradients to positive values only
    - Applying row normalization to the residual rollout
    - Supporting both L1 and L2 gradient importance metrics

    Parameters
    ----------
    alpha : float, optional
        Residual scaling added during rollout (default: 1.0).
    norm_type : str, optional
        Head importance metric; either ``'l1'`` or ``'l2'`` (default: 'l1').
    """

    def __init__(self, alpha: float = 1.0, norm_type: str = "l1") -> None:
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
        """Compute a normalized GMARv2 heatmap with ReLU-clamped gradients.

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

        model.zero_grad()

        # Backprop on the chosen class to populate gradients on attention
        target_logit = logits[0, pred_class]
        target_logit.backward(retain_graph=True)

        weighted_attns: List[torch.Tensor] = []

        for attn in attn_weights:
            grad = attn.grad
            if grad is None:
                raise RuntimeError(
                    "attn.grad is None — ensure `.retain_grad()` was called on attention tensors"
                )

            # Clamp gradients to positive values and weight attention
            pos_grad = grad.clamp(min=0)
            weighted_grad = pos_grad * attn

            # Compute head importance based on norm_type
            if self.norm_type == "l1":
                head_importance = pos_grad.abs().sum(dim=(-1, -2))
            else:  # l2
                head_importance = (pos_grad ** 2).sum(dim=(-1, -2)).sqrt()

            # Normalize head weights per layer
            head_weights = head_importance / head_importance.sum(dim=-1, keepdim=True)
            head_weights = head_weights.view(1, -1, 1, 1)

            # Collapse head dimension and keep the [1, N, N] attention matrix
            A_weighted = (weighted_grad * head_weights).sum(dim=1)
            weighted_attns.append(A_weighted)

        # Row-normalized residual rollout
        device = attn_weights[0].device
        N = weighted_attns[0].size(-1)
        rollout = torch.eye(N, device=device)

        for A in weighted_attns:
            A_residual = A + self.alpha * torch.eye(N, device=device)
            A_residual = A_residual / A_residual.sum(dim=-1, keepdim=True)
            rollout = rollout @ A_residual

        # Influence of class token on patches (exclude cls token at index 0)
        cls_influence = rollout[0, 0, 1:]
        side_len = int(cls_influence.numel() ** 0.5)
        cls_map = cls_influence.reshape(side_len, side_len).cpu().detach()

        # Safe normalization to [0, 1]
        mn = cls_map.min()
        mx = cls_map.max()
        if (mx - mn) <= 0:
            return torch.zeros_like(cls_map)
        cls_map = (cls_map - mn) / (mx - mn + 1e-8)
        return cls_map

    def plot_on_image(
        self,
        cls_map: torch.Tensor,
        original_image: Image.Image,
        cmap: str = "jet",
        alpha: float = 0.5,
    ) -> None:
        """Display the heatmap overlaid on the original image.

        Parameters
        ----------
        cls_map
            Heatmap tensor of shape [H, W] (typically 14×14).
        original_image
            Original input image (PIL.Image).
        cmap
            Matplotlib colormap name (default: 'jet').
        alpha
            Opacity of heatmap overlay (default: 0.5).
        """
        heatmap = np.array(cls_map)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = np.array(
            heatmap_img.resize(original_image.size, resample=Image.BILINEAR)
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(original_image)
        plt.imshow(heatmap_resized, cmap=cmap, alpha=alpha)
        plt.axis("off")
        plt.show()


    def save_overlay(
        self,
        cls_map: torch.Tensor,
        original_image: Image.Image,
        original_image_path: Union[str, Path],
    ) -> Path:
        """Save the heatmap overlay next to the original image and return path.

        The saved filename pattern is ``{original_stem}_gmarv2.png`` and the file
        is written into the same directory as ``original_image_path``.

        Parameters
        ----------
        cls_map
            Heatmap tensor of shape [H, W].
        original_image
            Original input image (PIL.Image).
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

        output_path = out_dir / f"{out_path.stem}_gmarv2.png"

        W, H = original_image.size
        heatmap = np.array(cls_map)

        hm_min, hm_max = heatmap.min(), heatmap.max()
        if (hm_max - hm_min) <= 0:
            heatmap_resized = np.zeros((H, W), dtype=np.uint8)
        else:
            heatmap_norm = (heatmap - hm_min) / (hm_max - hm_min + 1e-8)
            heatmap_img = Image.fromarray((heatmap_norm * 255).astype(np.uint8))
            heatmap_resized = np.array(
                heatmap_img.resize((W, H), resample=Image.BILINEAR)
            )

        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(original_image)
        ax.imshow(heatmap_resized, cmap="jet", alpha=0.5)
        ax.axis("off")

        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return output_path
    
        
