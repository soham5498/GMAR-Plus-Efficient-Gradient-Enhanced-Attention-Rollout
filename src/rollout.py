"""Attention Rollout: Classical Attention Aggregation for ViT Explainability

This module implements the classical Attention Rollout method, which computes
the influence of the [CLS] token on image patches by recursively multiplying
average attention matrices across all transformer layers.
"""

from pathlib import Path
from typing import List, Union
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


__all__ = ["AttentionRollout"]


class AttentionRollout:
    """Compute and save attention rollout heatmaps.

    This method recursively multiplies average attention matrices across all
    layers, optionally adding residual connections, to compute the cumulative
    influence of the [CLS] token on image patches.

    Parameters
    ----------
    add_residual : bool, optional
        Whether to add identity (residual) connections to attention matrices
        during rollout (default: True).
    """

    def __init__(self, add_residual: bool = True) -> None:
        self.add_residual = add_residual

    def compute_rollout(self, attn_weights: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention rollout by recursively multiplying attention maps.

        Starting with an identity matrix, multiplies attention matrices across
        all layers. Optionally applies row normalization with residual connections.

        Parameters
        ----------
        attn_weights
            List of attention tensors from ViT blocks. Each tensor has shape
            [B, H, N, N] where B is batch size, H is number of heads, and N is
            the number of tokens (1 + num_patches).

        Returns
        -------
        torch.Tensor
            Final 2D normalized attention map (excluding CLS token), shape [S, S]
            where S² = N - 1.
        """
        if not attn_weights:
            raise ValueError("attn_weights must be non-empty")

        device = attn_weights[0].device
        N = attn_weights[0].size(-1)
        rollout = torch.eye(N, device=device)

        for A in attn_weights:
            A_mean = A.mean(dim=1)

            if self.add_residual:
                A_res = A_mean + torch.eye(N, device=device)
                A_res = A_res / A_res.sum(dim=-1, keepdim=True)
            else:
                A_res = A_mean

            rollout = rollout @ A_res

        # Extract [CLS] token influence on patches (exclude CLS itself)
        cls_influence = rollout[0, 0, 1:]
        side_len = int(cls_influence.numel() ** 0.5)
        if side_len * side_len != cls_influence.numel():
            raise ValueError(
                f"Patch count {cls_influence.numel()} is not a perfect square"
            )

        cls_map = cls_influence.reshape(side_len, side_len).cpu().detach()

        # Normalize to [0, 1]
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
        """Save the heatmap overlay next to the original image and return path.

        The saved filename pattern is ``{original_stem}_rollout.png`` and the file
        is written into the same directory as ``original_image_path``.

        Parameters
        ----------
        cls_map: Heatmap tensor of shape [H, W].
        original_image: Original input image (PIL.Image).
        original_image_path: Path to the original image file.

        Returns
        -------
        Path
            Path where the overlay was saved.
        """
        out_path = Path(original_image_path)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        output_path = out_dir / f"{out_path.stem}_rollout.png"

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
        